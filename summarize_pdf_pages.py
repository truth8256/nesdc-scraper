"""
summarize_pdf_pages.py
======================
PDF의 각 페이지에서 상단 텍스트와 표 일부를 요약하여 CSV로 저장.
사용자가 Excel에서 Ctrl+F로 검색하여 원하는 페이지를 찾을 수 있게 함.

Usage:
    python summarize_pdf_pages.py --id-range 15300 15322
    python summarize_pdf_pages.py --id-list 15310 15320 15330
"""

import os
import sys
import io
import warnings
import argparse
import pandas as pd
import pdfplumber

# UTF-8 출력 설정 (unbuffered)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# 경고 숨기기
warnings.filterwarnings("ignore", message=".*gray non-stroke color.*")


def is_encoding_error(text: str, threshold=0.3) -> bool:
    """
    텍스트가 인코딩 오류를 포함하는지 확인.

    Parameters
    ----------
    text : str
        확인할 텍스트
    threshold : float
        오류 문자 비율 임계값 (기본 30%)

    Returns
    -------
    bool
        인코딩 오류 여부
    """
    if not text or len(text) < 10:
        return False

    # 인코딩 오류 지표: �, �, 깨진 문자 등
    error_chars = text.count('�') + text.count('�')

    # 한글이 거의 없고 알 수 없는 문자가 많은 경우
    # (정상적인 한글은 유니코드 범위 0xAC00-0xD7A3)
    korean_chars = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7A3)
    total_chars = len([c for c in text if c.strip()])

    if total_chars == 0:
        return False

    # 오류 문자가 많거나, 한글이 거의 없는 경우
    error_ratio = error_chars / total_chars
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0

    # 한글 문서인데 한글이 10% 미만이면 인코딩 오류로 판단
    if error_ratio > threshold or (total_chars > 50 and korean_ratio < 0.1):
        return True

    return False


def get_target_pdf(dir_path: str) -> str:
    """폴더에서 가장 큰 PDF 파일 찾기."""
    if not os.path.exists(dir_path):
        return None

    pdfs = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".pdf")
    ]

    if not pdfs:
        return None

    # 가장 큰 PDF 선택
    return max(pdfs, key=os.path.getsize)


def summarize_pdf_pages(pdf_path, merge_tables=True, max_lines=5, max_table_rows=3):
    """
    PDF 한 개 파일의 각 페이지에서 상단 텍스트와 표 일부를 요약.

    Parameters
    ----------
    pdf_path : str
        요약할 PDF 파일 경로
    merge_tables : bool, default=True
        True면 한 페이지 내의 모든 테이블을 병합하여 한 덩어리로 요약.
        False면 첫 번째 테이블만 요약.
    max_lines : int, default=5
        각 페이지에서 텍스트로 가져올 최대 줄 수.
    max_table_rows : int, default=3
        각 테이블(혹은 병합된 테이블)에서 가져올 최대 행 수.

    Returns
    -------
    list[dict]
        각 페이지의 요약 정보
    """
    summaries = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            # ① 텍스트 추출
            text = page.extract_text() or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            head_text = " / ".join(lines[:max_lines])

            # 인코딩 오류 체크 및 대체
            if is_encoding_error(head_text):
                head_text = "[ENCODING_ERROR] Text extraction failed due to encoding issues"

            # ② 테이블 추출
            tables = page.extract_tables()
            table_text = ""

            if tables:
                merged = []
                if merge_tables:
                    # 페이지 내 모든 테이블 병합
                    for t in tables:
                        merged.extend(t[:max_table_rows])
                else:
                    # 첫 번째 테이블만
                    merged = tables[0][:max_table_rows]

                # 각 행을 안전하게 문자열화
                table_text = " / ".join(
                    [
                        " | ".join([str(x) if x is not None else "" for x in row])
                        for row in merged
                        if any(row)
                    ]
                )

                # 테이블도 인코딩 오류 체크
                if is_encoding_error(table_text):
                    table_text = "[ENCODING_ERROR] Table extraction failed due to encoding issues"

            # ③ 요약 저장
            summaries.append({
                "page": i,
                "text_head": head_text[:400],   # 텍스트 너무 길면 자름
                "table_head": table_text[:400], # 테이블 내용도 400자 제한
            })

    return summaries


def summarize_folders(
    base_folder,
    out_path="pdf_page_summary_all.csv",
    id_range=None,
    id_list=None,
):
    """
    base_folder의 하위 폴더들을 순회하며:
      - 폴더명이 숫자인 경우 id_range나 id_list로 필터링 가능
      - 각 폴더에서 가장 큰 PDF 선택 후 요약
      - 폴더명(id)을 첫 열에 기록

    Parameters
    ----------
    base_folder : str
        상위 폴더 경로 (예: "data/raw")
    out_path : str
        출력 CSV 파일명
    id_range : tuple[int, int], optional
        (start, end) 범위로 필터링
    id_list : list[int], optional
        특정 ID 리스트로 필터링

    Returns
    -------
    pd.DataFrame
        전체 요약 결과
    """
    all_rows = []

    # 순회 시작
    folders = sorted(os.listdir(base_folder))

    for folder_name in folders:
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 폴더명이 숫자로 변환 가능한 경우만 처리
        try:
            folder_id = int(folder_name)
        except ValueError:
            continue

        # 범위 혹은 리스트 필터링
        if id_range:
            start, end = id_range
            if not (start <= folder_id <= end):
                continue
        if id_list and folder_id not in id_list:
            continue

        # PDF 찾기
        largest_pdf = get_target_pdf(folder_path)

        if not largest_pdf:
            print(f"⚠️  {folder_name}: PDF 없음")
            continue

        print(f"📂 {folder_name}: {os.path.basename(largest_pdf)}")

        try:
            summaries = summarize_pdf_pages(largest_pdf)
            for summary in summaries:
                summary["id"] = folder_name
                summary["pdf_name"] = os.path.basename(largest_pdf)
            all_rows.extend(summaries)
            print(f"   ✅ {len(summaries)} 페이지 요약 완료")
        except Exception as e:
            print(f"   ❌ 오류: {e}")

    # 저장
    if all_rows:
        # 열 순서: id, pdf_name, page, text_head, table_head
        df = pd.DataFrame(all_rows)
        df = df[["id", "pdf_name", "page", "text_head", "table_head"]]
        df.to_csv(out_path, index=False, encoding="utf-8-sig", escapechar='\\')
        print(f"\n💾 저장 완료: {out_path}")
        print(f"📊 총 {len(df)}개 페이지 요약")
        return df
    else:
        print("❌ 처리할 PDF가 없습니다.")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF 페이지 요약 생성")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="PDF가 있는 상위 폴더 (기본값: script_dir/data/raw)"
    )
    parser.add_argument(
        "--output",
        default="pdf_page_summary.csv",
        help="출력 CSV 파일명 (기본값: pdf_page_summary.csv)"
    )
    parser.add_argument(
        "--id-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="폴더 ID 범위 (예: --id-range 15300 15322)"
    )
    parser.add_argument(
        "--id-list",
        nargs='+',
        type=int,
        help="특정 폴더 ID 리스트 (예: --id-list 15310 15320)"
    )

    args = parser.parse_args()

    # 기본 경로 설정
    if args.base_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.base_dir = os.path.join(script_dir, "data", "raw")

    print(f"🔍 대상 폴더: {args.base_dir}")

    if args.id_range:
        print(f"📋 범위: {args.id_range[0]} ~ {args.id_range[1]}")
        id_range = tuple(args.id_range)
    else:
        id_range = None

    if args.id_list:
        print(f"📋 ID 리스트: {args.id_list}")

    # 실행
    df = summarize_folders(
        args.base_dir,
        out_path=args.output,
        id_range=id_range,
        id_list=args.id_list
    )

    if not df.empty:
        print(f"\n✨ 완료! {args.output} 파일을 Excel로 열어서 검색하세요.")
        print(f"   예: Ctrl+F로 '정당지지도' 검색 → 페이지 번호 확인")
