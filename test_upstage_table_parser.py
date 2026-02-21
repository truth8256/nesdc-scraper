"""
test_upstage_table_parser.py
=============================
업스테이지(Upstage) Document Parse API를 활용하여
PDF의 특정 페이지에서 표를 파싱하는 테스트 코드.

API 문서: https://developers.upstage.ai/docs/apis/document-parse
엔드포인트: https://api.upstage.ai/v1/document-digitization

동작 방식:
  1. PDF의 특정 페이지를 이미지(PNG)로 변환 (pdf2image)
  2. 변환된 이미지를 Upstage Document Parse API에 전송
  3. 응답에서 표(table) 요소를 추출
  4. HTML 형태의 표를 pandas DataFrame으로 변환
  5. JSON으로 저장

필요 패키지:
  pip install requests pdf2image pandas beautifulsoup4
  (Poppler도 필요: https://github.com/oschwartz10612/poppler-windows/releases)

사용법:
  python test_upstage_table_parser.py --page 12
  python test_upstage_table_parser.py --page 12 --ocr-mode enhanced
  python test_upstage_table_parser.py --page 12 --compare   # 모든 모드 비교

가능한 OCR 모드:
  auto      : 페이지를 자동 분류 후 standard/enhanced 적용 (기본값)
  standard  : 텍스트 중심 문서, 단순 표 처리
  enhanced  : 복잡한 표·차트·이미지 처리에 최적화된 Vision LM 사용
  force     : 강제 OCR (스캔본, 이미지 PDF에 유리)
"""

import os
import io
import json
import argparse
import tempfile
from pathlib import Path

import requests
import pandas as pd

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
UPSTAGE_API_KEY = "up_JJHX7JDeALql0mSSfjPj8g5XFjC5y"
UPSTAGE_API_URL = "https://api.upstage.ai/v1/document-digitization"

DEFAULT_PDF = (
    r"D:\working\polldata\nesdc_scraper\data\raw\15380"
    r"\3.통계표(260213)_MBC_2026년+설날특집+정치·사회현안+여론조사(2차)(15일보도).pdf"
)
DEFAULT_OUTPUT_DIR = r"D:\working\polldata\nesdc_scraper\data\upstage_output"


# ─────────────────────────────────────────────
# 1. PDF 페이지 → 이미지 변환
# ─────────────────────────────────────────────
def pdf_page_to_image_bytes(pdf_path: str, page_number: int, dpi: int = 200) -> bytes:
    """
    PDF의 특정 페이지를 PNG 바이트로 변환.
    Upstage API에 이미지로 전송하기 위해 사용.
    """
    from pdf2image import convert_from_path

    print(f"[PDF→이미지] 페이지 {page_number} 변환 중 (DPI={dpi})...")
    pages = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_number,
        last_page=page_number,
        fmt="png",
        # Poppler 경로가 필요한 경우:
        # poppler_path=r"C:\poppler\bin",
    )
    if not pages:
        raise ValueError(f"페이지 {page_number}를 변환하지 못했습니다.")

    buf = io.BytesIO()
    pages[0].save(buf, format="PNG")
    buf.seek(0)
    image_bytes = buf.read()
    print(f"  → 이미지 크기: {pages[0].size}, 바이트: {len(image_bytes):,}")
    return image_bytes


# ─────────────────────────────────────────────
# 2. Upstage Document Parse API 호출
# ─────────────────────────────────────────────
def call_upstage_api(
    image_bytes: bytes,
    filename: str = "page.png",
    output_formats: list = None,
    ocr_mode: str = "auto",
) -> dict:
    """
    Upstage Document Parse API를 호출하여 구조화된 결과를 반환.

    Parameters
    ----------
    image_bytes  : 이미지 바이트 (PNG/JPEG)
    filename     : 업로드 파일명
    output_formats : ["html", "markdown", "text"] 중 선택 (기본: ["html", "markdown"])
    ocr_mode     : "auto" | "force" (기본: "auto")

    Returns
    -------
    dict : Upstage API 응답 JSON
    """
    if output_formats is None:
        output_formats = ["html", "markdown", "text"]

    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
    }

    # output_formats을 쿼리 파라미터나 form-data로 전달
    files = {
        "document": (filename, image_bytes, "image/png"),
    }
    data = {
        "ocr": ocr_mode,
        "output_formats": json.dumps(output_formats),
        "model": "document-parse",
    }

    print(f"[API 호출] {UPSTAGE_API_URL}")
    print(f"  → output_formats={output_formats}, ocr={ocr_mode}")

    response = requests.post(
        UPSTAGE_API_URL,
        headers=headers,
        files=files,
        data=data,
        timeout=120,
    )

    print(f"  → HTTP Status: {response.status_code}")

    if response.status_code != 200:
        print(f"  [오류] 응답 내용:\n{response.text[:1000]}")
        response.raise_for_status()

    return response.json()


# ─────────────────────────────────────────────
# 3. HTML 표 → DataFrame 변환
# ─────────────────────────────────────────────
def html_tables_to_dataframes(html_content: str) -> list[pd.DataFrame]:
    """
    HTML 문자열에서 <table> 태그를 찾아 DataFrame 목록으로 변환.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 패키지를 설치하세요: pip install beautifulsoup4")

    # pandas의 read_html은 lxml/html5lib가 필요하므로 BeautifulSoup로 직접 파싱
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    print(f"  → HTML에서 감지된 <table> 수: {len(tables)}")

    dataframes = []
    for i, table in enumerate(tables):
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                # colspan/rowspan 무시하고 텍스트만 추출
                text = td.get_text(separator=" ", strip=True)
                cells.append(text)
            if cells:
                rows.append(cells)

        if not rows:
            continue

        # 첫 행을 헤더로 사용
        headers = rows[0]
        data_rows = rows[1:]

        # 열 수 통일
        num_cols = len(headers)
        padded = []
        for row in data_rows:
            if len(row) < num_cols:
                row = row + [""] * (num_cols - len(row))
            elif len(row) > num_cols:
                row = row[:num_cols]
            padded.append(row)

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

        df = pd.DataFrame(padded, columns=unique_headers)
        dataframes.append(df)
        print(f"    [표 {i+1}] {len(df)} 행 × {len(df.columns)} 열")

    return dataframes


# ─────────────────────────────────────────────
# 4. 응답 결과 출력 및 저장
# ─────────────────────────────────────────────
def print_api_response_summary(response: dict):
    """API 응답 구조를 요약 출력"""
    print("\n[응답 구조]")
    print(f"  키 목록: {list(response.keys())}")

    # 주요 필드
    if "pages" in response:
        print(f"  pages 수: {len(response['pages'])}")
        for i, page in enumerate(response["pages"]):
            print(f"  [page {i}] 키: {list(page.keys())}")
            if "content" in page:
                content = page["content"]
                for fmt, val in content.items():
                    snippet = str(val)[:100].replace("\n", " ")
                    print(f"    [{fmt}] {snippet}...")

    if "api" in response:
        print(f"  api: {response['api']}")
    if "model" in response:
        print(f"  model: {response['model']}")


def save_results(
    response: dict,
    dataframes: list,
    output_dir: str,
    page_number: int,
    pdf_path: str,
    ocr_mode: str = "auto",
):
    """결과를 JSON, CSV, 마크다운으로 저장 (파일명에 모드 포함)"""
    os.makedirs(output_dir, exist_ok=True)
    base = f"{Path(pdf_path).stem}_page{page_number}_{ocr_mode}"

    # 1) 원본 API 응답 JSON 저장
    raw_path = os.path.join(output_dir, f"{base}_response.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    print(f"  [저장] 원본 응답 → {raw_path}")

    # 2) DataFrame별 CSV 저장
    for i, df in enumerate(dataframes):
        csv_path = os.path.join(output_dir, f"{base}_table{i+1}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  [저장] 표 {i+1} CSV → {csv_path}")

    # 3) 마크다운 추출 저장
    md_content = extract_content(response, fmt="markdown")
    if md_content:
        md_path = os.path.join(output_dir, f"{base}_content.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  [저장] 마크다운 → {md_path}")

    # 4) DataFrame JSON 저장
    tables_data = [
        {"table_index": i + 1, "columns": df.columns.tolist(), "records": df.to_dict(orient="records")}
        for i, df in enumerate(dataframes)
    ]
    json_path = os.path.join(output_dir, f"{base}_tables.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tables_data, f, ensure_ascii=False, indent=2)
    print(f"  [저장] 표 JSON → {json_path}")

    return json_path


def extract_content(response: dict, fmt: str = "html") -> str:
    """
    API 응답에서 특정 포맷의 content를 추출.
    응답 구조: response["content"][fmt] 또는 response["pages"][0]["content"][fmt]
    """
    # 최상위 content 필드
    if "content" in response:
        content = response["content"]
        if isinstance(content, dict) and fmt in content:
            return content[fmt]
        elif isinstance(content, str):
            return content

    # pages 배열 방식
    if "pages" in response:
        parts = []
        for page in response["pages"]:
            c = page.get("content", {})
            if isinstance(c, dict) and fmt in c:
                parts.append(c[fmt])
            elif isinstance(c, str):
                parts.append(c)
        return "\n\n".join(parts)

    return ""


# ─────────────────────────────────────────────
# 5. 메인 파이프라인
# ─────────────────────────────────────────────
def parse_table_with_upstage(
    pdf_path: str,
    page_number: int,
    dpi: int = 200,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ocr_mode: str = "auto",
) -> dict:
    """
    PDF 특정 페이지를 Upstage API로 파싱하여 표를 추출하는 메인 함수.

    ocr_mode 옵션:
      - "auto"     : 자동 분류 후 standard/enhanced 각각 적용
      - "standard" : 텍스트 중심 단순 표 처리
      - "enhanced" : 복잡한 표/차트에 Vision LM 사용 (복잡 통계표 권장)
      - "force"    : 강제 OCR (스캔 PDF에 유리)
    """
    print(f"\n{'='*60}")
    print(f"[Upstage 파이프라인] {Path(pdf_path).name} / 페이지 {page_number} / 모드: {ocr_mode}")
    print(f"{'='*60}")

    # Step 1: PDF → 이미지
    filename = f"{Path(pdf_path).stem}_p{page_number}.png"
    image_bytes = pdf_page_to_image_bytes(pdf_path, page_number, dpi=dpi)

    # Step 2: API 호출
    response = call_upstage_api(
        image_bytes,
        filename=filename,
        output_formats=["html", "markdown", "text"],
        ocr_mode=ocr_mode,
    )

    # Step 3: 응답 요약 출력
    print_api_response_summary(response)

    # Step 4: HTML → DataFrame 변환
    print("\n[표 추출]")
    html_content = extract_content(response, "html")
    if not html_content:
        print("  [경고] HTML content가 비어 있습니다.")
        md = extract_content(response, "markdown")
        if md:
            print("\n[마크다운 내용]\n" + md[:2000])
        dataframes = []
    else:
        print(f"  HTML 길이: {len(html_content)} 문자")
        dataframes = html_tables_to_dataframes(html_content)

    # Step 5: 결과 출력
    print(f"\n[추출된 표: {len(dataframes)}개]")
    for i, df in enumerate(dataframes):
        print(f"\n  ── 표 {i+1} ({len(df)}행 × {len(df.columns)}열) ──")
        print(df.to_string(index=False))

    # Step 6: 마크다운 출력 (보조)
    md_content = extract_content(response, "markdown")
    if md_content:
        print(f"\n[마크다운 미리보기 (처음 1500자)]\n{'─'*40}")
        print(md_content[:1500])

    # Step 7: 저장 (모드를 파일명에 포함)
    print(f"\n[결과 저장]")
    json_path = save_results(
        response, dataframes, output_dir, page_number, pdf_path,
        ocr_mode=ocr_mode
    )

    return {
        "page": page_number,
        "pdf": pdf_path,
        "ocr_mode": ocr_mode,
        "response": response,
        "dataframes": dataframes,
        "output_json": json_path,
    }


def compare_all_modes(
    pdf_path: str,
    page_number: int,
    dpi: int = 200,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    standard / enhanced / auto 세 가지 모드를 모두 실행하여 결과를 비교.
    이미지 변환은 한 번만 수행.
    """
    from pdf2image import convert_from_path

    print(f"\n{'='*60}")
    print(f"[모드 비교 테스트] {Path(pdf_path).name} / 페이지 {page_number}")
    print(f"{'='*60}")

    # 이미지 한 번만 변환
    filename = f"{Path(pdf_path).stem}_p{page_number}.png"
    image_bytes = pdf_page_to_image_bytes(pdf_path, page_number, dpi=dpi)

    modes = ["standard", "enhanced", "auto"]
    results = {}
    summary_rows = []

    for mode in modes:
        print(f"\n{'─'*50}")
        print(f"[모드: {mode}] API 호출 중...")
        print(f"{'─'*50}")
        try:
            response = call_upstage_api(
                image_bytes,
                filename=filename,
                output_formats=["html", "markdown"],
                ocr_mode=mode,
            )

            # HTML → 표 추출
            html = extract_content(response, "html")
            dfs = html_tables_to_dataframes(html) if html else []
            md = extract_content(response, "markdown")

            # 저장
            save_results(response, dfs, output_dir, page_number, pdf_path, ocr_mode=mode)

            summary_rows.append({
                "모드": mode,
                "감지된 표 수": len(dfs),
                "총 행 수": sum(len(d) for d in dfs),
                "마크다운 길이(자)": len(md),
                "상태": "성공",
            })
            results[mode] = {"response": response, "dataframes": dfs}
        except Exception as e:
            summary_rows.append({"모드": mode, "감지된 표 수": 0, "총 행 수": 0,
                                  "마크다운 길이(자)": 0, "상태": f"오류: {e}"})
            results[mode] = {"error": str(e)}

    # 비교 요약 출력
    print(f"\n{'='*60}")
    print("[모드별 비교 결과]")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary_rows).set_index("모드")
    print(summary_df.to_string())

    # 비교 요약 저장
    compare_path = os.path.join(
        output_dir,
        f"{Path(pdf_path).stem}_page{page_number}_mode_compare.csv"
    )
    summary_df.to_csv(compare_path, encoding="utf-8-sig")
    print(f"\n  비교 결과 저장 → {compare_path}")

    return results


# ─────────────────────────────────────────────
# 6. CLI 진입점
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upstage Document Parse API 기반 PDF 표 파싱 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드 설명:
  auto      페이지를 자동 분류 후 standard/enhanced 적용 (기본값)
  standard  텍스트 중심 단순 표 처리
  enhanced  복잡한 표·차트에 최적화된 Vision LM 사용 (복잡 통계표 권장)
  force     강제 OCR (스캔 PDF에 유리)

예시:
  python test_upstage_table_parser.py --page 12 --ocr-mode enhanced
  python test_upstage_table_parser.py --page 12 --compare
        """
    )
    parser.add_argument(
        "--pdf", type=str, default=DEFAULT_PDF,
        help="PDF 파일 경로 (기본값: MBC 2026 설날특집 통계표)",
    )
    parser.add_argument(
        "--page", type=int, default=12,
        help="파싱할 페이지 번호 (1-indexed, 기본값: 12)",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="이미지 변환 DPI (기본값: 200)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--ocr-mode", type=str, default="auto",
        choices=["auto", "standard", "enhanced", "force"],
        help="OCR 처리 모드 (기본값: auto)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="standard / enhanced / auto 세 모드를 모두 실행하여 비교",
    )
    parser.add_argument(
        "--api-key", type=str, default=UPSTAGE_API_KEY,
        help="Upstage API 키 (기본값: 코드 내 설정값)",
    )

    args = parser.parse_args()
    UPSTAGE_API_KEY = args.api_key

    if not os.path.isfile(args.pdf):
        print(f"[오류] PDF 파일을 찾을 수 없습니다: {args.pdf}")
        exit(1)

    if args.compare:
        # ─── 모든 모드 비교 실행 ───
        compare_all_modes(
            pdf_path=args.pdf,
            page_number=args.page,
            dpi=args.dpi,
            output_dir=args.output_dir,
        )
    else:
        # ─── 단일 모드 실행 ───
        result = parse_table_with_upstage(
            pdf_path=args.pdf,
            page_number=args.page,
            dpi=args.dpi,
            output_dir=args.output_dir,
            ocr_mode=args.ocr_mode,
        )

        dfs = result["dataframes"]
        print(f"\n{'='*60}")
        print(f"[최종 결과] 모드={args.ocr_mode}, 추출된 표 수: {len(dfs)}")
        for i, df in enumerate(dfs):
            print(f"  표 {i+1}: {len(df)}행 × {len(df.columns)}열")
            print(f"  컬럼: {df.columns.tolist()}")
        print(f"  결과 저장 경로: {result['output_json']}")
