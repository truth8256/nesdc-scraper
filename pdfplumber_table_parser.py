"""
pdfplumber_table_parser.py
===========================
pdfplumber의 extract_text()를 이용한 통계표 파서.
PDF에 텍스트 레이어가 내장되어 있으므로 OCR 불필요.
"""
import os, sys, re, json
import pdfplumber
import pandas as pd


def parse_stat_table(pdf_path: str, page_num: int, output_dir: str) -> pd.DataFrame:
    """
    통계표 PDF의 특정 페이지를 파싱하여 DataFrame으로 반환.
    
    Parameters
    ----------
    pdf_path  : PDF 파일 경로
    page_num  : 페이지 번호 (1-indexed)
    output_dir: 결과 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        text = page.extract_text()

    if not text:
        print("[오류] 텍스트를 추출할 수 없습니다.")
        return pd.DataFrame()

    lines = text.strip().split("\n")
    print(f"[pdfplumber] 페이지 {page_num}: {len(lines)}줄 추출")

    # ─── 1. 헤더/데이터 영역 분리 ───
    # 헤더 구분선(--------) 위치 찾기
    separator_lines = []
    for i, line in enumerate(lines):
        if re.match(r"^[\s\-]+$", line) and len(line.strip()) > 20:
            separator_lines.append(i)

    if len(separator_lines) < 2:
        print(f"[경고] 구분선이 충분하지 않음 ({len(separator_lines)}개). 전체 파싱 시도.")
        data_start = 0
        data_end = len(lines)
    else:
        # 마지막 헤더 구분선 이후가 데이터 영역
        # 보통: 구분선0 = 제목 하단, 구분선1 = 컬럼 헤더 하단
        data_start = separator_lines[-2] + 1  # 헤더 구분선 다음
        data_end = separator_lines[-1] if len(separator_lines) > 2 else len(lines)
        
        # 실제 데이터가 시작되는 줄 찾기 (숫자 포함)
        for i in range(len(separator_lines)):
            line_after = lines[separator_lines[i] + 1] if separator_lines[i] + 1 < len(lines) else ""
            if re.search(r"\(\d+\)", line_after):
                data_start = separator_lines[i] + 1
                data_end = separator_lines[i + 1] if i + 1 < len(separator_lines) else len(lines)
                break

    print(f"  데이터 영역: 줄 {data_start}~{data_end}")

    # ─── 2. 데이터 행 파싱 ───
    # 패턴: 레이블 (사례수) 숫자1 숫자2 ... 숫자N (가중사례수)
    data_rows = []
    header_line = None

    for i in range(data_start, data_end):
        line = lines[i].strip()
        if not line or re.match(r"^[\s\-]+$", line):
            continue
        if line.startswith("◈"):
            # 대분류 구분 행 (예: ◈ 성 ◈)
            continue

        # (숫자) 패턴으로 사례수와 데이터 분리
        # 형태: "전 체 (1000) 19 33 24 20 5 52 44 5 (1000)"
        m = re.match(
            r"^(.+?)\s+\((\d+)\)\s+([\d\s]+)\s+\((\d+)\)\s*$",
            line
        )
        if m:
            label = m.group(1).replace(" ", "")  # 글자 사이 공백 제거
            sample_n = int(m.group(2))
            values_str = m.group(3).strip()
            weighted_n = int(m.group(4))
            
            values = [int(v) for v in values_str.split()]
            row = [label, sample_n] + values + [weighted_n]
            data_rows.append(row)
        else:
            # 매치 안 되면 원본 저장
            print(f"  [미매치] 줄 {i}: {line[:60]}")

    if not data_rows:
        print("[오류] 파싱된 데이터 행이 없습니다.")
        return pd.DataFrame()

    # ─── 3. 컬럼명 결정 ───
    # 첫 데이터 행의 값 수로 컬럼 수 결정
    n_values = len(data_rows[0]) - 3  # label, sample_n, ..values.., weighted_n
    
    # 기본 컬럼명 (텍스트에서 추출 시도)
    col_names = ["구분", "사례수"]
    
    # 헤더에서 컬럼명 추출 시도
    for i in range(data_start):
        line = lines[i].strip()
        if "매우" in line and "효과" in line:
            # 이 줄에서 응답 선택지를 포함
            pass

    # 일반적인 응답 패턴 사용
    if n_values == 8:
        col_names += [
            "매우효과있음", "어느정도효과", "별로효과없음", "전혀효과없음",
            "모름/무응답",
            "긍정평가", "부정평가", "모름/무응답(종합)"
        ]
    else:
        col_names += [f"값{j+1}" for j in range(n_values)]
    
    col_names.append("가중사례수")

    # ─── 4. DataFrame 생성 ───
    df = pd.DataFrame(data_rows, columns=col_names)
    
    print(f"\n[결과] {len(df)} 행 × {len(df.columns)} 열")
    print(df.to_string())

    # ─── 5. 저장 ───
    csv_path = os.path.join(output_dir, f"page{page_num}_table.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV 저장 → {csv_path}")

    json_path = os.path.join(output_dir, f"page{page_num}_table.json")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"  JSON 저장 → {json_path}")

    return df


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    PDF_PATH = (
        r"D:\working\polldata\nesdc_scraper\data\raw\15380"
        r"\3.통계표(260213)_MBC_2026년+설날특집+정치·사회현안+여론조사(2차)(15일보도).pdf"
    )
    OUTPUT_DIR = r"D:\working\polldata\nesdc_scraper\data\pdfplumber_output"

    df = parse_stat_table(PDF_PATH, page_num=12, output_dir=OUTPUT_DIR)
