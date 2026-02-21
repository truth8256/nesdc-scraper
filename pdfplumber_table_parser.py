"""
pdfplumber_table_parser.py
===========================
pdfplumber의 extract_text()를 이용한 통계표 파서.
PDF에 텍스트 레이어가 내장되어 있으므로 OCR 불필요.

Phase 2.1에서 table_parser.py와 동일한 인터페이스로 리팩토링됨.
"""
import os
import sys
import re
import json
import pdfplumber
import pandas as pd
from schema import infer_group, convert_flat_to_standard, validate_schema
from validator import validate_table


def get_target_pdf(dir_path):
    """
    Identify the target PDF in the directory.
    Strategy: Select the LARGEST PDF file (likely the Result Report).
    """
    if not os.path.exists(dir_path):
        return None

    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]

    if not files:
        return None

    # Get full paths and sizes
    file_info = []
    for f in files:
        full_path = os.path.join(dir_path, f)
        size = os.path.getsize(full_path)
        file_info.append((f, size, full_path))

    # Sort by size (descending) - LARGEST first
    file_info.sort(key=lambda x: x[1], reverse=True)

    return file_info[0][2]  # Return full path of largest file


def extract_question_from_text(text_lines, data_start_idx):
    """
    페이지 상단에서 질문 텍스트 추출.

    Parameters
    ----------
    text_lines : list[str]
        페이지의 모든 텍스트 줄
    data_start_idx : int
        데이터 영역 시작 인덱스

    Returns
    -------
    str  질문 텍스트 (없으면 빈 문자열)
    """
    # 데이터 영역 이전의 텍스트에서 "Q숫자.", "SQ숫자.", "문" 등으로 시작하는 줄 찾기
    question_pattern = re.compile(r'^(Q\d+\.|SQ\d+\.|문\s*\d+\.)')

    for i in range(min(data_start_idx, len(text_lines))):
        line = text_lines[i].strip()
        if question_pattern.match(line):
            # 여러 줄에 걸쳐 있을 수 있으므로 다음 줄도 포함
            question = line
            for j in range(i + 1, min(i + 5, data_start_idx)):
                next_line = text_lines[j].strip()
                # 구분선이나 다른 패턴이 나오면 중단
                if re.match(r"^[\s\-]+$", next_line) or question_pattern.match(next_line):
                    break
                if next_line and not next_line.startswith("◈"):
                    question += " " + next_line
            return question.strip()

    return ""


def parse_single_page(pdf_path, page_num):
    """
    PDF의 특정 페이지를 파싱하여 flat DataFrame 반환.

    Parameters
    ----------
    pdf_path : str
        PDF 파일 경로
    page_num : int
        페이지 번호 (1-indexed)

    Returns
    -------
    tuple: (pd.DataFrame, str)  # (데이터프레임, 질문 텍스트)
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        text = page.extract_text()

    if not text:
        print(f"[경고] 페이지 {page_num}: 텍스트를 추출할 수 없습니다.")
        return pd.DataFrame(), ""

    lines = text.strip().split("\n")
    print(f"[pdfplumber] 페이지 {page_num}: {len(lines)}줄 추출")

    # ─── 1. 헤더/데이터 영역 분리 ───
    separator_lines = []
    for i, line in enumerate(lines):
        if re.match(r"^[\s\-]+$", line) and len(line.strip()) > 20:
            separator_lines.append(i)

    if len(separator_lines) < 2:
        print(f"[경고] 구분선이 충분하지 않음 ({len(separator_lines)}개). 전체 파싱 시도.")
        data_start = 0
        data_end = len(lines)
    else:
        # 실제 데이터가 시작되는 줄 찾기 (숫자 포함)
        data_start = separator_lines[-2] + 1
        data_end = separator_lines[-1] if len(separator_lines) > 2 else len(lines)

        for i in range(len(separator_lines)):
            line_after = lines[separator_lines[i] + 1] if separator_lines[i] + 1 < len(lines) else ""
            if re.search(r"\(\d+\)", line_after):
                data_start = separator_lines[i] + 1
                data_end = separator_lines[i + 1] if i + 1 < len(separator_lines) else len(lines)
                break

    print(f"  데이터 영역: 줄 {data_start}~{data_end}")

    # ─── 2. 질문 추출 ───
    question = extract_question_from_text(lines, data_start)
    if question:
        print(f"  질문: {question[:100]}...")

    # ─── 3. 컬럼명 추출 ───
    # 데이터 시작 이전 영역에서 컬럼명 찾기
    header_candidates = []
    for i in range(max(0, data_start - 10), data_start):
        line = lines[i].strip()
        # 일반적인 응답 선택지 패턴
        if any(keyword in line for keyword in ["매우", "어느정도", "전혀", "모름", "무응답", "긍정", "부정"]):
            # 공백으로 분리된 토큰들을 컬럼 후보로 추가
            tokens = line.split()
            header_candidates.extend(tokens)

    # ─── 4. 데이터 행 파싱 ───
    data_rows = []

    for i in range(data_start, data_end):
        line = lines[i].strip()
        if not line or re.match(r"^[\s\-]+$", line):
            continue
        if line.startswith("◈"):
            # 대분류 구분 행
            continue

        # 패턴: "레이블 (사례수) 숫자1 숫자2 ... 숫자N (가중사례수)"
        m = re.match(
            r"^(.+?)\s+\((\d+)\)\s+([\d\s]+)\s+\((\d+)\)\s*$",
            line
        )
        if m:
            label = m.group(1).replace(" ", "")
            sample_n = int(m.group(2))
            values_str = m.group(3).strip()
            weighted_n = int(m.group(4))

            values = [int(v) for v in values_str.split()]
            data_rows.append({
                "label": label,
                "n_raw": sample_n,
                "values": values,
                "n_weighted": weighted_n
            })
        else:
            # 패턴 매치 실패 - 다른 포맷 시도
            # "레이블 (사례수) 값1 값2 ..."
            m2 = re.match(r"^(.+?)\s+\((\d+)\)\s+([\d\s\.]+)$", line)
            if m2:
                label = m2.group(1).replace(" ", "")
                sample_n = int(m2.group(2))
                values_str = m2.group(3).strip()
                values = [float(v) for v in values_str.split()]
                data_rows.append({
                    "label": label,
                    "n_raw": sample_n,
                    "values": values,
                    "n_weighted": None
                })

    if not data_rows:
        print(f"[경고] 페이지 {page_num}: 파싱된 데이터 행이 없습니다.")
        return pd.DataFrame(), question

    # ─── 5. DataFrame 생성 ───
    # 값 개수가 일정한지 확인
    value_counts = [len(row["values"]) for row in data_rows]
    if len(set(value_counts)) > 1:
        print(f"[경고] 행마다 값 개수가 다릅니다: {set(value_counts)}")
        # 가장 흔한 개수 사용
        from collections import Counter
        common_count = Counter(value_counts).most_common(1)[0][0]
        data_rows = [row for row in data_rows if len(row["values"]) == common_count]

    n_values = len(data_rows[0]["values"])

    # 컬럼명 결정
    if len(header_candidates) >= n_values:
        value_col_names = header_candidates[:n_values]
    else:
        # 기본 컬럼명
        value_col_names = [f"응답{i+1}" for i in range(n_values)]

    # DataFrame 구성
    flat_data = []
    for row in data_rows:
        entry = {"구분": row["label"], "사례수": row["n_raw"]}
        for i, val in enumerate(row["values"]):
            entry[value_col_names[i]] = val
        if row["n_weighted"] is not None:
            entry["가중사례수"] = row["n_weighted"]
        flat_data.append(entry)

    df = pd.DataFrame(flat_data)
    print(f"[결과] {len(df)} 행 × {len(df.columns)} 열")

    return df, question


def parse_survey_table(folder_name, page_numbers, force_llm=False):
    """
    Extract tables from specific pages of the largest PDF in the folder.

    Parameters
    ----------
    folder_name : str
        폴더명 (예: "15380")
    page_numbers : list[int]
        파싱할 페이지 번호 리스트 (1-indexed)
    force_llm : bool
        미사용 (pdfplumber에는 LLM fallback 없음, 호환성 위해 유지)

    Returns
    -------
    None (파일로 저장)
    """
    # 현재 스크립트 디렉토리 기준 상대 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "data", "raw")
    dir_path = os.path.join(base_dir, str(folder_name))

    target_pdf = get_target_pdf(dir_path)
    if not target_pdf:
        print(f"No PDF found in {dir_path}")
        return

    print(f"Targeting file: {target_pdf}")

    extracted_tables = []

    # 페이지 정렬 및 중복 제거
    pages_sorted = sorted(list(set(page_numbers)))

    for page_num in pages_sorted:
        print(f"\nProcessing page {page_num}...")

        try:
            df, question = parse_single_page(target_pdf, page_num)

            if df.empty:
                print(f"  - No data extracted from page {page_num}")
                continue

            # ─── Validation ───
            page_info = {'page': page_num}
            report = validate_table(df, page_info)
            score = report.get('row_validity_rate', 0)
            status = report.get('validity', 'Unknown')

            print(f"  - Validation: {status} ({score:.1f}%)")

            # ─── 표준 스키마 변환 ───
            flat_data = df.to_dict(orient='records')

            standard_table = convert_flat_to_standard(
                flat_data,
                poll_id=str(folder_name),
                page=page_num,
                keyword="",  # 자동 추론은 Phase 3에서 구현
                question=question,
                method="pdfplumber_v1.0"
            )

            # ─── 스키마 검증 ───
            schema_issues = validate_schema(standard_table)
            if schema_issues:
                print(f"  - Schema validation warnings:")
                for issue in schema_issues[:3]:  # 최대 3개만 출력
                    print(f"    • {issue}")

            extracted_tables.append(standard_table)

        except Exception as e:
            print(f"  - Error processing page {page_num}: {e}")
            import traceback
            traceback.print_exc()

    # ─── 저장 ───
    output_dir = os.path.join(script_dir, "data", "parsed_tables")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{folder_name}_pdfplumber.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_tables, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(extracted_tables)} tables to {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import io
    import argparse

    # UTF-8 출력 설정
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extract tables from PDF survey results using pdfplumber.")
    parser.add_argument("--folder", type=str, required=True, help="Folder name (e.g., 15312)")
    parser.add_argument("--pages", type=int, nargs="+", required=True, help="List of page numbers to extract (e.g., 25 27 41)")
    parser.add_argument("--force-llm", action="store_true", help="Ignored (compatibility with table_parser.py)")

    args = parser.parse_args()

    print(f"Processing Folder: {args.folder}, Pages: {args.pages}")
    parse_survey_table(args.folder, args.pages, force_llm=args.force_llm)
