"""
custom_table_parser.py
=======================
투영 프로파일(Projection Profile) 기반 자체 표 인식기.

대상 표의 구조적 특성:
  - 행 간격(Row Gap): ≈ 0 (글자가 위아래로 붙어 있음)
  - 열 간격(Column Gap): 행 간격보다 훨씬 넓고 명확함
  - 언어: 한국어 + 숫자 (폰트 일정)

알고리즘:
  1. 이진화: 픽셀 < threshold → 글자(1), 그 외 → 배경(0)
  2. H-투영: 행별 픽셀 합 → 텍스트 행 밴드 탐지
  3. V-투영: 열별 픽셀 합 → 텍스트 열 밴드 탐지 (행 밴드 내에서)
  4. 셀 크롭 + OCR (RapidOCR, ONNX 기반) → 텍스트 추출
  5. DataFrame 조립 + CSV/JSON 저장
"""

import os
import io
import json
import numpy as np
from pathlib import Path
from typing import Optional
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
DEFAULT_PDF = (
    r"D:\working\polldata\nesdc_scraper\data\raw\15380"
    r"\3.통계표(260213)_MBC_2026년+설날특집+정치·사회현안+여론조사(2차)(15일보도).pdf"
)
DEFAULT_OUTPUT_DIR = r"D:\working\polldata\nesdc_scraper\data\custom_output"

# ─────────────────────────────────────────────
# 1. 이미지 로드 & 이진화
# ─────────────────────────────────────────────

def load_image_from_pdf(pdf_path: str, page: int, dpi: int = 200) -> Image.Image:
    """PDF 특정 페이지를 PIL Image로 변환."""
    from pdf2image import convert_from_path
    print(f"[PDF→이미지] 페이지 {page} (DPI={dpi})...")
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=page, last_page=page, fmt="png")
    if not pages:
        raise ValueError(f"페이지 {page} 변환 실패")
    print(f"  크기: {pages[0].size}")
    return pages[0]


def load_image_from_file(image_path: str) -> Image.Image:
    """파일에서 PIL Image 로드."""
    return Image.open(image_path)


def binarize(pil_img: Image.Image, threshold: int = 200) -> np.ndarray:
    """
    이미지를 이진화.
    반환: shape=(H, W), dtype=uint8, 글자=1, 배경=0
    """
    gray = np.array(pil_img.convert("L"))
    binary = (gray < threshold).astype(np.uint8)
    return binary


# ─────────────────────────────────────────────
# 2. 행 밴드 탐지 (H-Projection)
# ─────────────────────────────────────────────

def detect_row_bands(
    binary: np.ndarray,
    min_gap: int = 3,
    h_threshold_ratio: float = 0.02,
    min_height: int = 5,
) -> list[tuple[int, int]]:
    """
    수평 투영 프로파일로 텍스트 행 밴드를 탐지.

    Parameters
    ----------
    binary            : 이진화 배열 (H×W)
    min_gap           : 행과 행 사이 빈 픽셀 최소 수 (이 이상이면 행 분리)
    h_threshold_ratio : 행 인정 임계값 (max H-proj의 비율)
    min_height        : 최소 행 높이 (픽셀)

    Returns
    -------
    [(y_start, y_end), ...] 형태의 행 밴드 목록
    """
    h_proj = binary.sum(axis=1)  # shape: (H,)
    threshold = max(3, h_proj.max() * h_threshold_ratio)
    text_mask = (h_proj > threshold).astype(int)

    # 연속 구간 탐지
    diff = np.diff(np.concatenate([[0], text_mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    bands = []
    for s, e in zip(starts, ends):
        if e - s + 1 >= min_height:
            bands.append((int(s), int(e)))

    print(f"  [H-투영] 탐지된 행 밴드: {len(bands)}개")
    return bands


# ─────────────────────────────────────────────
# 3. 열 밴드 탐지 (V-Projection)
# ─────────────────────────────────────────────

def detect_col_bands_global(
    binary: np.ndarray,
    min_gap: int = 10,
    v_threshold_ratio: float = 0.02,
    min_width: int = 3,
) -> list[tuple[int, int]]:
    """
    전체 이미지의 수직 투영으로 열 밴드를 탐지.
    표의 열 구조가 행마다 일정하다면 이 방법이 안정적.

    Returns
    -------
    [(x_start, x_end), ...] 형태의 열 밴드 목록
    """
    v_proj = binary.sum(axis=0)  # shape: (W,)
    threshold = max(2, v_proj.max() * v_threshold_ratio)
    col_mask = (v_proj > threshold).astype(int)

    diff = np.diff(np.concatenate([[0], col_mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    # min_gap보다 작은 간격으로 분리된 열 밴드 병합
    merged = []
    for s, e in zip(starts, ends):
        if e - s + 1 < min_width:
            continue
        if merged and s - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    result = [(int(s), int(e)) for s, e in merged]
    print(f"  [V-투영] 탐지된 열 밴드: {len(result)}개")
    return result


def detect_col_bands_in_rows(
    binary: np.ndarray,
    row_bands: list[tuple[int, int]],
    min_gap: int = 10,
    v_threshold_ratio: float = 0.01,
    min_width: int = 3,
    sample_rows: int = 20,
) -> list[tuple[int, int]]:
    """
    여러 행 밴드의 V-투영을 합산하여 열 밴드를 탐지.
    일부 행만 샘플링하여 계산 (빠름).
    """
    H, W = binary.shape
    v_accum = np.zeros(W, dtype=np.int64)
    binary64 = binary.astype(np.int64)

    # 샘플 행 선택 (균등 분포)
    if len(row_bands) > sample_rows:
        step = len(row_bands) // sample_rows
        sample = row_bands[::step][:sample_rows]
    else:
        sample = row_bands

    for (ys, ye) in sample:
        v_accum += binary64[int(ys):int(ye)+1, :].sum(axis=0)

    threshold = max(2, v_accum.max() * v_threshold_ratio)
    col_mask = (v_accum > threshold).astype(int)

    diff = np.diff(np.concatenate([[0], col_mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    merged = []
    for s, e in zip(starts, ends):
        if e - s + 1 < min_width:
            continue
        if merged and s - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    result = [(int(s), int(e)) for s, e in merged]
    print(f"  [V-투영/행합산] 탐지된 열 밴드: {len(result)}개")
    return result


def detect_col_bands_per_row(
    binary: np.ndarray,
    row_band: tuple,
    min_gap: int = 5,
) -> list:
    """
    단일 행 밴드에서 V-투영으로 셀 x 범위를 탐지.
    min_gap 픽셀 이상의 빈 공간이 나오면 셀을 분리.
    """
    ys, ye = int(row_band[0]), int(row_band[1])
    v = binary.astype(np.int64)[ys:ye+1, :].sum(axis=0)
    col_mask = (v > 0).astype(int)
    diff = np.diff(np.concatenate([[0], col_mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    merged = []
    for s, e in zip(starts, ends):
        w = int(e) - int(s) + 1
        if w < 2:  # 1px 노이즈 제거
            continue
        if merged and int(s) - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append([int(s), int(e)])
    return [(s, e) for s, e in merged]


def build_global_col_bands(
    binary: np.ndarray,
    row_bands: list,
    min_gap: int = 8,
    min_density: float = 0.3,
    min_width: int = 8,
) -> list:
    """
    모든 행에서 탐지된 x 좌표를 조합하여
    오버래핑되는 구간을 클러스터링하여 글로벌 열 범위를 반환.

    min_density : 열 범위를 확인할 수 있는 행 비율(0~1)
    min_width   : 최소 열 너비 (px, 너무 작은 노이즈 제거)
    """
    W = binary.shape[1]
    x_hit = np.zeros(W, dtype=np.float32)
    n_rows = len(row_bands)

    for rb in row_bands:
        cells = detect_col_bands_per_row(binary, rb, min_gap=min_gap)
        for (xs, xe) in cells:
            x_hit[xs:xe+1] += 1.0

    x_density = x_hit / n_rows
    active = (x_density >= min_density).astype(int)

    diff = np.diff(np.concatenate([[0], active, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    merged = []
    for s, e in zip(starts, ends):
        if e - s + 1 < min_width:
            continue
        if merged and int(s) - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append([int(s), int(e)])

    result = [(int(s), int(e)) for s, e in merged]
    print(f"  [클러스터 V-투영] 열 밴드 {len(result)}개 (density>={min_density}, width>={min_width})")
    return result





# ─────────────────────────────────────────────
# 4. 셀 크롭 + OCR
# ─────────────────────────────────────────────

_ocr_instance = None

def get_ocr():
    """
    RapidOCR 싱글턴.
    pip install rapidocr-onnxruntime 으로 설치.
    PaddleOCR보다 10~50배 빠름 (ONNX 추론 기반).
    """
    global _ocr_instance
    if _ocr_instance is None:
        from rapidocr_onnxruntime import RapidOCR
        print("  [RapidOCR] 초기화...")
        _ocr_instance = RapidOCR()
        print("  [RapidOCR] 준비 완료")
    return _ocr_instance


def ocr_image(pil_img: Image.Image) -> str:
    """
    PIL 이미지 → 텍스트 (RapidOCR).
    빈 이미지이면 빈 문자열 반환.
    """
    # 글자 픽셀이 거의 없으면 스킵
    gray = np.array(pil_img.convert("L"))
    if (gray < 200).sum() < 3:
        return ""

    arr = np.array(pil_img.convert("RGB"))
    ocr = get_ocr()
    try:
        result, _ = ocr(arr)
    except Exception as e:
        return f"[err:{e}]"

    if not result:
        return ""

    # result: [ [box, text, score], ... ]
    texts = []
    for item in result:
        if item and len(item) >= 2:
            txt = item[1] if isinstance(item[1], str) else str(item[1])
            texts.append(txt.strip())

    return " ".join(texts).strip()


def crop_and_pad(pil_img: Image.Image, y1: int, y2: int, x1: int, x2: int, pad: int = 4) -> Image.Image:
    """이미지에서 셀 영역을 패딩 포함하여 크롭."""
    W, H = pil_img.size
    y1 = max(0, y1 - pad)
    y2 = min(H - 1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(W - 1, x2 + pad)
    return pil_img.crop((x1, y1, x2 + 1, y2 + 1))


# ─────────────────────────────────────────────
# 5. 전체 테이블 OCR
# ─────────────────────────────────────────────

def extract_table(
    pil_img: Image.Image,
    row_bands: list,
    col_bands: list,
    pad: int = 4,
    label_x_end: int = 0,
    verbose: bool = True,
) -> list[list[str]]:
    """
    행/열 밴드를 기반으로 모든 셀을 크롭하고 OCR 수행.

    Parameters
    ----------
    label_x_end : 0보다 크면 x=0~label_x_end 범위를 하나의 '레이블 열'로 처리하고
                  col_bands에서 x <= label_x_end 인 열들을 제외.
                  보통 숫자 데이터 시작 x 좌표보다 조금 작게 설정.

    Returns
    -------
    2D 문자열 배열 texts[row_idx][col_idx]
    """
    # 레이블 열 분리
    if label_x_end > 0:
        data_cols = [(xs, xe) for (xs, xe) in col_bands if xs > label_x_end]
        effective_cols = [(-1, label_x_end)] + data_cols  # (-1, x_end) = 레이블 마커
    else:
        effective_cols = col_bands
        data_cols = col_bands

    total = len(row_bands) * len(effective_cols)
    print(f"  [셀 OCR] {len(row_bands)}행 × {len(effective_cols)}열 = {total}셀 처리 중...")
    if label_x_end > 0:
        print(f"    레이블 열: x=0~{label_x_end}, 데이터 열: {len(data_cols)}개")

    texts = []
    for r_idx, (ys, ye) in enumerate(row_bands):
        row_texts = []
        for c_idx, (xs, xe) in enumerate(effective_cols):
            if xs == -1:
                # 레이블 열: x=0~label_x_end 전체를 크롭
                cell = crop_and_pad(pil_img, ys, ye, 0, xe, pad=pad)
            else:
                cell = crop_and_pad(pil_img, ys, ye, xs, xe, pad=pad)
            text = ocr_image(cell)
            row_texts.append(text)
            if verbose and (r_idx * len(effective_cols) + c_idx) % 30 == 0:
                progress = (r_idx * len(effective_cols) + c_idx + 1) / total * 100
                print(f"    {progress:.0f}% (행 {r_idx+1}/{len(row_bands)}, 열 {c_idx+1}): '{text[:15]}'")
        texts.append(row_texts)

    return texts


def build_dataframe(
    texts: list[list[str]],
    label_ffill_cols: int = 5,
) -> pd.DataFrame:
    """
    2D 텍스트 배열 → DataFrame.

    Parameters
    ----------
    label_ffill_cols : 좌측 N열에 ffill 적용 (rowspan으로 인해 비어있는 행 레이블 채우기)
                       0이면 ffill 안 함.
    """
    if not texts:
        return pd.DataFrame()

    # 컬럼 수 통일
    max_cols = max(len(row) for row in texts)
    padded = [row + [""] * (max_cols - len(row)) for row in texts]

    # 첫 행을 헤더로
    headers = padded[0]
    seen: dict = {}
    unique = []
    for h in headers:
        key = h or "col"
        if key in seen:
            seen[key] += 1
            unique.append(f"{key}_{seen[key]}")
        else:
            seen[key] = 0
            unique.append(key)

    df = pd.DataFrame(padded[1:], columns=unique)

    # 행 레이블 열 ffill: 빈 문자열 → NaN → ffill → 빈 문자열 복원
    if label_ffill_cols > 0:
        label_cols = list(df.columns[:label_ffill_cols])
        df[label_cols] = (
            df[label_cols]
            .replace("", pd.NA)
            .ffill()
            .fillna("")
        )

    return df


# ─────────────────────────────────────────────
# 6. 시각화 (디버그)
# ─────────────────────────────────────────────

def visualize(
    pil_img: Image.Image,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    texts: Optional[list[list[str]]] = None,
    out_path: str = "debug.png",
    scale: float = 0.5,
):
    """
    탐지된 행/열 밴드를 원본 이미지에 오버레이하여 저장.
    - 행 밴드: 빨간 수평선
    - 열 밴드: 파란 수직선
    - 셀 텍스트: 초록색 텍스트 (texts 제공 시)
    """
    viz = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", viz.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 행 밴드 (빨간 반투명 영역)
    for ys, ye in row_bands:
        draw.rectangle([(0, ys), (pil_img.width, ye)], fill=(255, 0, 0, 25))
        draw.line([(0, ys), (pil_img.width, ys)], fill=(255, 0, 0, 180), width=1)
        draw.line([(0, ye), (pil_img.width, ye)], fill=(255, 100, 100, 120), width=1)

    # 열 밴드 (파란 반투명 영역)
    for xs, xe in col_bands:
        draw.rectangle([(xs, 0), (xe, pil_img.height)], fill=(0, 0, 255, 15))
        draw.line([(xs, 0), (xs, pil_img.height)], fill=(0, 0, 255, 180), width=1)
        draw.line([(xe, 0), (xe, pil_img.height)], fill=(0, 100, 255, 120), width=1)

    viz = Image.alpha_composite(viz, overlay).convert("RGB")

    # 셀 텍스트 오버레이
    if texts:
        draw2 = ImageDraw.Draw(viz)
        try:
            # 한국어 폰트 시도
            font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 10)
        except Exception:
            font = ImageFont.load_default()

        for r_idx, (ys, ye) in enumerate(row_bands):
            if r_idx >= len(texts):
                break
            for c_idx, (xs, xe) in enumerate(col_bands):
                if c_idx >= len(texts[r_idx]):
                    break
                txt = texts[r_idx][c_idx]
                if txt:
                    # 셀 중앙에 텍스트
                    tx = xs + 2
                    ty = ys + 1
                    draw2.text((tx, ty), txt[:12], fill=(0, 150, 0), font=font)

    # 스케일 다운하여 저장
    if scale != 1.0:
        new_w = int(viz.width * scale)
        new_h = int(viz.height * scale)
        viz = viz.resize((new_w, new_h), Image.LANCZOS)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    viz.save(out_path)
    print(f"  [시각화] 저장 → {out_path}")


# ─────────────────────────────────────────────
# 7. 결과 저장
# ─────────────────────────────────────────────

def save_results(
    df: pd.DataFrame,
    texts: list[list[str]],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    output_dir: str,
    label: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, f"{label}_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [저장] CSV → {csv_path}")

    # JSON (원시 2D 배열)
    json_path = os.path.join(output_dir, f"{label}_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "row_bands": row_bands,
                "col_bands": col_bands,
                "texts": texts,
                "dataframe": df.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  [저장] JSON → {json_path}")

    return csv_path, json_path


# ─────────────────────────────────────────────
# 8. 전체 파이프라인
# ─────────────────────────────────────────────

def parse_table(
    pdf_path: str = DEFAULT_PDF,
    page: int = 12,
    dpi: int = 200,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    # 이진화
    binarize_threshold: int = 200,
    # 행 탐지
    row_min_gap: int = 3,
    row_h_threshold: float = 0.02,
    row_min_height: int = 5,
    # 열 탐지
    col_min_gap: int = 10,
    col_v_threshold: float = 0.01,
    # OCR
    cell_pad: int = 4,
    # 출력
    debug_image: bool = True,
    debug_scale: float = 0.5,
    image_path: Optional[str] = None,
) -> dict:
    """
    PDF 특정 페이지에서 투영 프로파일로 표를 파싱하는 전체 파이프라인.

    Parameters
    ----------
    pdf_path        : 입력 PDF 경로
    page            : 파싱할 페이지 (1-indexed)
    image_path      : PDF 대신 이미지 직접 지정 (옵션)
    """
    print(f"\n{'='*60}")
    print(f"[자체 표 인식기] {Path(pdf_path).name if not image_path else image_path} / 페이지 {page}")
    print(f"{'='*60}")

    label = f"page{page}"
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: 이미지 로드 ──
    if image_path:
        pil_img = load_image_from_file(image_path)
        print(f"  이미지 로드: {pil_img.size}")
    else:
        pil_img = load_image_from_pdf(pdf_path, page, dpi=dpi)

    # ── Step 2: 이진화 ──
    binary = binarize(pil_img, threshold=binarize_threshold)
    print(f"  이진화 완료 (threshold={binarize_threshold}), 글자 픽셀: {binary.sum():,}")

    # ── Step 3: 행 밴드 탐지 ──
    print("\n[행 탐지]")
    row_bands = detect_row_bands(
        binary,
        min_gap=row_min_gap,
        h_threshold_ratio=row_h_threshold,
        min_height=row_min_height,
    )
    if not row_bands:
        print("  [오류] 행을 탐지하지 못했습니다.")
        return {}

    # ── Step 4: 열 밴드 탐지 (행별 V-투영 클러스터링) ──
    print("\n[열 탐지]")
    col_bands = build_global_col_bands(
        binary,
        row_bands,
        min_gap=col_min_gap,
        min_density=0.25,
    )
    if not col_bands:
        print("  [오류] 열을 탐지하지 못했습니다.")
        return {}

    print(f"\n  → 표 크기 추정: {len(row_bands)} 행 × {len(col_bands)} 열")

    # ── Step 5: 시각화 (OCR 전 확인용) ──
    if debug_image:
        debug_no_text_path = os.path.join(output_dir, f"{label}_debug_bands.png")
        visualize(pil_img, row_bands, col_bands, texts=None, out_path=debug_no_text_path, scale=debug_scale)

    # ── Step 6: 셀 OCR ──
    print("\n[셀 OCR 수행]")
    # label_x_end: 레이블 영역 경계 (이 x 이하는 통합 레이블 열로 처리)
    # 숫자 데이터 열이 x~600 이후에 시작하므로 580으로 설정
    label_x_end = 580
    texts = extract_table(pil_img, row_bands, col_bands, pad=cell_pad, label_x_end=label_x_end)

    # ── Step 7: DataFrame 조립 ──
    df = build_dataframe(texts)
    print(f"\n[DataFrame] {len(df)} 행 × {len(df.columns)} 열")
    print(df.head(5).to_string())

    # ── Step 8: 시각화 (텍스트 포함) ──
    if debug_image:
        debug_text_path = os.path.join(output_dir, f"{label}_debug_ocr.png")
        visualize(pil_img, row_bands, col_bands, texts=texts, out_path=debug_text_path, scale=debug_scale)

    # ── Step 9: 저장 ──
    print("\n[결과 저장]")
    csv_path, json_path = save_results(df, texts, row_bands, col_bands, output_dir, label)

    return {
        "page": page,
        "pil_img": pil_img,
        "binary": binary,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "texts": texts,
        "dataframe": df,
        "csv_path": csv_path,
        "json_path": json_path,
    }
