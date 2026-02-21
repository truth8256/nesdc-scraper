"""
test_paddle_table_parser.py
============================
PaddleOCR을 활용하여 PDF의 표를 파싱하는 테스트 코드.

동작 방식:
  1. pdf2image로 PDF 특정 페이지를 이미지로 변환
  2. PaddleOCR의 구조 분석(table recognition) 또는
     OCR 결과 boundingbox 기반으로 셀을 재조합하여 표 추출
  3. 결과를 pandas DataFrame 및 JSON으로 저장

필요 패키지 (Python 3.11 권장, PaddleOCR은 3.13 미지원):
  conda create -n paddle311 python=3.11 -y
  conda activate paddle311
  pip install paddlepaddle paddleocr pdf2image pandas opencv-python
  (Poppler도 필요: https://github.com/oschwartz10612/poppler-windows/releases)

실행 예시:
  conda run -n paddle311 python test_paddle_table_parser.py --pdf <경로> --page 1
"""

# ─────────────────────────────────────────────
# 환경변수는 반드시 import 전에 설정해야 합니다
# ─────────────────────────────────────────────
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"  # 모델 소스 연결 체크 비활성화

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# 1. PDF → 이미지 변환
# ─────────────────────────────────────────────
def pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 200) -> np.ndarray:
    """
    PDF의 특정 페이지(1-indexed)를 numpy 이미지 배열로 변환.
    pdf2image + Poppler가 필요합니다.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("pdf2image 패키지를 설치하세요: pip install pdf2image")

    print(f"[PDF→이미지] 페이지 {page_number} 변환 중 (DPI={dpi})...")
    pages = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_number,
        last_page=page_number,
        # Poppler 경로가 필요한 경우 아래를 설정하세요.
        # poppler_path=r"C:\poppler\bin",
    )
    if not pages:
        raise ValueError(f"페이지 {page_number}를 변환하지 못했습니다.")

    img = np.array(pages[0])
    print(f"  → 이미지 크기: {img.shape}")
    return img


# ─────────────────────────────────────────────
# 2. PaddleOCR 초기화 (싱글턴)
# ─────────────────────────────────────────────
_ocr_instance = None

def get_ocr(lang: str = "korean"):
    """
    PaddleOCR 인스턴스를 싱글턴으로 반환.
    lang 옵션: 'korean', 'ch', 'en' 등
    """
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("paddleocr 패키지를 설치하세요: pip install paddleocr paddlepaddle")

        print(f"[PaddleOCR] 모델 초기화 (lang={lang})...")
        # PaddleOCR 버전에 따라 지원하는 파라미터가 다르므로 순차 폴백
        init_configs = [
            # 3.x 최신: 보조 모델 비활성화로 빠른 초기화
            {
                "lang": lang,
                "use_doc_orientation_classify": False,  # 문서 방향 분류 모델 생략
                "use_doc_unwarping": False,             # 문서 펼침 모델 생략
                "use_textline_orientation": False,      # 텍스트라인 방향 분류 생략
            },
            # 폴백: 최소 파라미터
            {"lang": lang},
            # 구버전 호환: use_angle_cls, show_log 포함
            {"use_angle_cls": True, "lang": lang, "show_log": False},
        ]
        for config in init_configs:
            try:
                _ocr_instance = PaddleOCR(**config)
                print(f"  → 초기화 완료 (config={list(config.keys())})")
                break
            except (TypeError, ValueError) as e:
                print(f"  [폴백] {e}")
                _ocr_instance = None

        if _ocr_instance is None:
            raise RuntimeError("PaddleOCR 초기화에 실패했습니다.")
    return _ocr_instance


# ─────────────────────────────────────────────
# 3. OCR 실행 및 결과 파싱
# ─────────────────────────────────────────────
def run_ocr(image: np.ndarray) -> list:
    """
    이미지에 PaddleOCR을 실행하여 텍스트 블록 목록을 반환.

    PaddleOCR 2.x: ocr() → list[list[tuple(box, (text, conf))]]
    PaddleOCR 3.x: ocr() → OCRResult 객체 (이터러블, .boxes/.txts/.scores 속성 존재)

    반환 형식:
      [{"text": str, "confidence": float,
        "x0": int, "y0": int, "x1": int, "y1": int}, ...]
    """
    ocr = get_ocr()
    result = ocr.ocr(image)

    blocks = []

    # ── PaddleOCR 3.x: OCRResult 객체 ──────────────────────
    # OCRResult는 인덱싱(result[0])이 안 되고 속성을 사용함
    if hasattr(result, "boxes") and hasattr(result, "txts"):
        boxes = result.boxes or []
        txts  = result.txts  or []
        scores = result.scores or [1.0] * len(txts)
        for box, text, conf in zip(boxes, txts, scores):
            if text is None:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            blocks.append({
                "text": str(text).strip(),
                "confidence": round(float(conf), 3),
                "x0": int(min(xs)),
                "y0": int(min(ys)),
                "x1": int(max(xs)),
                "y1": int(max(ys)),
            })

    # ── PaddleOCR 3.x: OCRResult가 리스트 형태일 때 ────────
    elif isinstance(result, list) and result and isinstance(result[0], list):
        for page_result in result:
            if not page_result:
                continue
            for item in page_result:
                # item = (box, (text, conf)) 또는 다른 형태
                if not item:
                    continue
                try:
                    if len(item) == 2 and isinstance(item[1], (tuple, list)):
                        box, rec = item
                        text, conf = rec[0], rec[1]
                    else:
                        # {'transcription': ..., 'points': ..., 'score': ...}
                        text = item.get("transcription", "")
                        conf = item.get("score", 1.0)
                        box  = item.get("points", [[0,0],[0,0],[0,0],[0,0]])
                except Exception:
                    continue
                if not text:
                    continue
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                blocks.append({
                    "text": str(text).strip(),
                    "confidence": round(float(conf), 3),
                    "x0": int(min(xs)),
                    "y0": int(min(ys)),
                    "x1": int(max(xs)),
                    "y1": int(max(ys)),
                })
    else:
        print(f"  [OCR] 알 수 없는 반환 형식: {type(result)} — 원본 덤프:")
        print(f"  {result}")

    if not blocks:
        print("  [OCR] 텍스트를 감지하지 못했습니다.")
    else:
        print(f"  [OCR] 감지된 텍스트 블록 수: {len(blocks)}")
    return blocks


# ─────────────────────────────────────────────
# 4. Bounding Box 기반 표 구조 재조합
# ─────────────────────────────────────────────
def cluster_by_rows(blocks: list[dict], row_tolerance: int = 10) -> list[list[dict]]:
    """
    Y좌표를 기준으로 텍스트 블록을 '행(row)' 단위로 클러스터링.
    row_tolerance: 같은 행으로 볼 Y 좌표 차이 허용치(px)
    """
    if not blocks:
        return []

    sorted_blocks = sorted(blocks, key=lambda b: (b["y0"], b["x0"]))
    rows = []
    current_row = [sorted_blocks[0]]
    current_y = sorted_blocks[0]["y0"]

    for block in sorted_blocks[1:]:
        if abs(block["y0"] - current_y) <= row_tolerance:
            current_row.append(block)
        else:
            rows.append(sorted(current_row, key=lambda b: b["x0"]))  # 행 내부는 X 순 정렬
            current_row = [block]
            current_y = block["y0"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x0"]))

    return rows


def assign_columns(rows: list[list[dict]], col_tolerance: int = 20) -> list[list[str]]:
    """
    행들의 X좌표를 분석해 컬럼 경계를 추정하고,
    각 행에 컬럼 인덱스를 할당하여 2D 리스트로 반환.
    """
    if not rows:
        return []

    # 모든 블록의 x0 수집 → 컬럼 경계 추정
    all_x0 = sorted(set(b["x0"] for row in rows for b in row))

    # 인접한 x0 값 병합 (같은 컬럼으로 간주)
    col_anchors = [all_x0[0]]
    for x in all_x0[1:]:
        if x - col_anchors[-1] > col_tolerance:
            col_anchors.append(x)

    num_cols = len(col_anchors)

    def find_col_idx(x0):
        """블록의 x0에 가장 가까운 컬럼 인덱스 반환"""
        diffs = [abs(x0 - anchor) for anchor in col_anchors]
        return diffs.index(min(diffs))

    table_2d = []
    for row in rows:
        row_cells = [""] * num_cols
        for block in row:
            col_idx = find_col_idx(block["x0"])
            # 동일 셀에 텍스트가 합쳐질 경우 공백으로 연결
            if row_cells[col_idx]:
                row_cells[col_idx] += " " + block["text"]
            else:
                row_cells[col_idx] = block["text"]
        table_2d.append(row_cells)

    return table_2d


def table_to_dataframe(table_2d: list[list[str]]) -> pd.DataFrame:
    """
    2D 리스트를 DataFrame으로 변환 (첫 행을 헤더로 사용).
    """
    if not table_2d:
        return pd.DataFrame()

    headers = table_2d[0]
    data_rows = table_2d[1:]

    # 헤더 중복 처리
    seen = {}
    unique_headers = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)

    # 열 수 불일치 보정
    num_cols = len(unique_headers)
    padded_rows = []
    for row in data_rows:
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        elif len(row) > num_cols:
            row = row[:num_cols]
        padded_rows.append(row)

    return pd.DataFrame(padded_rows, columns=unique_headers)


# ─────────────────────────────────────────────
# 5. 메인 파이프라인
# ─────────────────────────────────────────────
def parse_table_from_pdf(
    pdf_path: str,
    page_number: int,
    row_tolerance: int = 10,
    col_tolerance: int = 20,
    dpi: int = 200,
    output_dir: str = None,
    save_debug_image: bool = False,
) -> dict:
    """
    PDF 페이지에서 표를 추출하는 메인 함수.

    Returns:
        {
          "page": int,
          "pdf": str,
          "blocks": [...],       # 원본 OCR 결과
          "table_2d": [...],     # 재조합된 2D 표
          "dataframe": DataFrame,
          "output_json": str,    # 저장된 JSON 경로 (output_dir 지정 시)
        }
    """
    print(f"\n{'='*60}")
    print(f"[파이프라인 시작] {os.path.basename(pdf_path)} / 페이지 {page_number}")
    print(f"{'='*60}")

    # Step 1: PDF → 이미지
    image = pdf_page_to_image(pdf_path, page_number, dpi=dpi)

    # Step 2: OCR
    blocks = run_ocr(image)
    if not blocks:
        print("  [경고] OCR 결과가 없습니다. 이미지를 확인하세요.")
        return {"page": page_number, "pdf": pdf_path, "blocks": [], "table_2d": [], "dataframe": pd.DataFrame()}

    # Step 3: 행 클러스터링
    rows = cluster_by_rows(blocks, row_tolerance=row_tolerance)
    print(f"  [행 클러스터링] 감지된 행 수: {len(rows)}")

    # Step 4: 컬럼 할당 → 2D 표
    table_2d = assign_columns(rows, col_tolerance=col_tolerance)
    print(f"  [표 재조합] 행 x 열 = {len(table_2d)} x {len(table_2d[0]) if table_2d else 0}")

    # Step 5: DataFrame 변환
    df = table_to_dataframe(table_2d)
    print(f"\n[추출된 DataFrame]\n{df.to_string(index=False)}\n")

    # Step 6: 저장 (선택)
    result = {
        "page": page_number,
        "pdf": pdf_path,
        "blocks": blocks,
        "table_2d": table_2d,
        "dataframe": df,
        "output_json": None,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(pdf_path).stem
        out_path = os.path.join(output_dir, f"{base_name}_page{page_number}_paddle.json")
        save_data = {
            "page": page_number,
            "pdf": pdf_path,
            "table_2d": table_2d,
            "records": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"  [저장] JSON → {out_path}")
        result["output_json"] = out_path

    # Step 7: 디버그 이미지 저장 (선택)
    if save_debug_image and output_dir:
        try:
            import cv2
            debug_img = image.copy()
            for b in blocks:
                cv2.rectangle(debug_img, (b["x0"], b["y0"]), (b["x1"], b["y1"]), (0, 255, 0), 1)
                cv2.putText(debug_img, b["text"][:10], (b["x0"], b["y0"] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            debug_path = os.path.join(output_dir, f"{base_name}_page{page_number}_debug.jpg")
            cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
            print(f"  [디버그 이미지] → {debug_path}")
        except ImportError:
            print("  [디버그 이미지] opencv-python이 없어 건너뜁니다.")

    return result


# ─────────────────────────────────────────────
# 6. 단위 테스트 함수들
# ─────────────────────────────────────────────
def test_cluster_by_rows():
    """행 클러스터링 단위 테스트"""
    print("\n[테스트] cluster_by_rows")
    dummy_blocks = [
        {"text": "구분", "x0": 10, "y0": 50, "x1": 60, "y1": 70},
        {"text": "응답자수", "x0": 80, "y0": 52, "x1": 160, "y1": 72},
        {"text": "비율", "x0": 180, "y0": 51, "x1": 220, "y1": 71},
        {"text": "매우긍정", "x0": 10, "y0": 100, "x1": 80, "y1": 120},
        {"text": "150", "x0": 80, "y0": 102, "x1": 160, "y1": 122},
        {"text": "30.0%", "x0": 180, "y0": 101, "x1": 230, "y1": 121},
        {"text": "긍정", "x0": 10, "y0": 150, "x1": 50, "y1": 170},
        {"text": "200", "x0": 80, "y0": 152, "x1": 160, "y1": 172},
        {"text": "40.0%", "x0": 180, "y0": 151, "x1": 230, "y1": 171},
    ]
    rows = cluster_by_rows(dummy_blocks, row_tolerance=10)
    assert len(rows) == 3, f"행 수 오류: {len(rows)} (기대값: 3)"
    print(f"  → 행 수: {len(rows)} ✓")

    table_2d = assign_columns(rows, col_tolerance=20)
    df = table_to_dataframe(table_2d)
    print(f"  → DataFrame:\n{df.to_string(index=False)}")
    assert list(df.columns) == ["구분", "응답자수", "비율"], f"컬럼 오류: {df.columns}"
    assert len(df) == 2, f"데이터 행 수 오류: {len(df)}"
    print("  → 단위 테스트 통과 ✓\n")


def test_table_to_dataframe_empty():
    """빈 입력 테스트"""
    print("[테스트] table_to_dataframe (빈 입력)")
    df = table_to_dataframe([])
    assert df.empty
    print("  → 빈 DataFrame 반환 ✓\n")


def run_all_unit_tests():
    print("\n" + "="*60)
    print("단위 테스트 실행")
    print("="*60)
    test_cluster_by_rows()
    test_table_to_dataframe_empty()
    print("모든 단위 테스트 통과! ✓")


# ─────────────────────────────────────────────
# 7. CLI 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PaddleOCR 기반 PDF 표 파싱 테스트"
    )
    parser.add_argument(
        "--pdf", type=str,
        help="PDF 파일 경로 (미지정 시 단위 테스트만 실행)",
    )
    parser.add_argument(
        "--page", type=int, default=1,
        help="파싱할 페이지 번호 (1-indexed, 기본값: 1)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="이미지 변환 DPI (기본값: 200)",
    )
    parser.add_argument(
        "--row-tol", type=int, default=10,
        help="행 클러스터링 Y좌표 허용치(px) (기본값: 10)",
    )
    parser.add_argument(
        "--col-tol", type=int, default=20,
        help="컬럼 경계 X좌표 허용치(px) (기본값: 20)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=r"d:\working\polldata\nesdc_scraper\data\paddle_output",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--debug-image", action="store_true",
        help="OCR 결과를 표시한 디버그 이미지 저장 (opencv 필요)",
    )
    parser.add_argument(
        "--unit-test", action="store_true",
        help="단위 테스트만 실행",
    )

    args = parser.parse_args()

    # 단위 테스트
    run_all_unit_tests()

    # PDF 파싱 테스트
    if args.unit_test or not args.pdf:
        print("\n[안내] --pdf 옵션을 지정하면 실제 PDF에서 표를 추출합니다.")
        print("예시:")
        print(r'  python test_paddle_table_parser.py --pdf "d:\working\polldata\nesdc_scraper\data\raw\10\청송군여론조사보도.pdf" --page 1 --debug-image')
    else:
        if not os.path.isfile(args.pdf):
            print(f"[오류] PDF 파일을 찾을 수 없습니다: {args.pdf}")
            exit(1)

        result = parse_table_from_pdf(
            pdf_path=args.pdf,
            page_number=args.page,
            row_tolerance=args.row_tol,
            col_tolerance=args.col_tol,
            dpi=args.dpi,
            output_dir=args.output_dir,
            save_debug_image=args.debug_image,
        )

        df = result["dataframe"]
        print(f"\n[최종 결과] 행={len(df)}, 열={len(df.columns)}")
        if not df.empty:
            print(df.to_string(index=False))
