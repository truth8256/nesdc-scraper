"""
test_pdfplumber.py
==================
pdfplumber로 PDF 표를 직접 추출하는 테스트.
pdfplumber는 PDF의 텍스트 레이어를 직접 읽으므로 OCR 불필요, 매우 빠름.
"""
import os, sys, json
import pdfplumber
import pandas as pd

PDF_PATH = (
    r"D:\working\polldata\nesdc_scraper\data\raw\15380"
    r"\3.통계표(260213)_MBC_2026년+설날특집+정치·사회현안+여론조사(2차)(15일보도).pdf"
)
OUTPUT_DIR = r"D:\working\polldata\nesdc_scraper\data\pdfplumber_output"
PAGE_NUM = 12  # 1-indexed

os.makedirs(OUTPUT_DIR, exist_ok=True)

# stdout 인코딩 설정
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print(f"{'='*60}")
print(f"[pdfplumber 테스트] {os.path.basename(PDF_PATH)} / 페이지 {PAGE_NUM}")
print(f"{'='*60}")

with pdfplumber.open(PDF_PATH) as pdf:
    print(f"  PDF 총 페이지: {len(pdf.pages)}")
    page = pdf.pages[PAGE_NUM - 1]  # 0-indexed
    print(f"  페이지 크기: {page.width} x {page.height}")

    # ─── 1. 전체 텍스트 추출 ───
    text = page.extract_text()
    text_path = os.path.join(OUTPUT_DIR, f"page{PAGE_NUM}_text.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text or "")
    print(f"\n[전체 텍스트] 글자 수: {len(text or '')}")
    print(f"  저장 → {text_path}")
    if text:
        lines = text.strip().split("\n")
        print(f"  줄 수: {len(lines)}")
        print(f"  처음 5줄:")
        for line in lines[:5]:
            print(f"    {line}")

    # ─── 2. 표 추출 (기본 설정) ───
    print(f"\n{'─'*50}")
    print("[표 추출 - 기본 설정]")
    tables = page.extract_tables()
    print(f"  탐지된 표 수: {len(tables)}")
    for i, table in enumerate(tables):
        df = pd.DataFrame(table[1:], columns=table[0])
        csv_path = os.path.join(OUTPUT_DIR, f"page{PAGE_NUM}_table{i+1}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  표 {i+1}: {df.shape[0]} 행 × {df.shape[1]} 열 → {csv_path}")
        print(f"  컬럼: {list(df.columns)}")
        print(f"  처음 5행:")
        print(df.head().to_string(index=False))
        print()

    # ─── 3. 표 추출 (text_strategy 조정) ───
    print(f"\n{'─'*50}")
    print("[표 추출 - text_x_tolerance=5]")
    tables2 = page.extract_tables({
        "text_x_tolerance": 5,
        "text_y_tolerance": 3,
    })
    print(f"  탐지된 표 수: {len(tables2)}")
    for i, table in enumerate(tables2):
        df = pd.DataFrame(table[1:], columns=table[0])
        csv_path = os.path.join(OUTPUT_DIR, f"page{PAGE_NUM}_table{i+1}_tol5.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  표 {i+1}: {df.shape[0]} 행 × {df.shape[1]} 열 → {csv_path}")
        print(f"  컬럼: {list(df.columns)}")
        print(f"  처음 5행:")
        print(df.head().to_string(index=False))

    # ─── 4. 표 추출 (세로선 직접 지정 시도) ───
    print(f"\n{'─'*50}")
    print("[표 추출 - find_tables + 디버그]")
    found = page.find_tables()
    print(f"  find_tables 결과: {len(found)}개")
    for i, t in enumerate(found):
        print(f"  표 {i+1} bbox: {t.bbox}")
        df = pd.DataFrame(t.extract()[1:], columns=t.extract()[0])
        csv_path = os.path.join(OUTPUT_DIR, f"page{PAGE_NUM}_find{i+1}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"    {df.shape[0]} 행 × {df.shape[1]} 열 → {csv_path}")

    # ─── 5. 텍스트 words 추출 (위치 포함) ───
    print(f"\n{'─'*50}")
    print("[텍스트 words 추출 (위치 포함)]")
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    words_path = os.path.join(OUTPUT_DIR, f"page{PAGE_NUM}_words.json")
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)
    print(f"  총 words: {len(words)}개 → {words_path}")
    if words:
        print(f"  처음 10개:")
        for w in words[:10]:
            print(f"    x={w['x0']:.0f}~{w['x1']:.0f}, y={w['top']:.0f}~{w['bottom']:.0f}: '{w['text']}'")

print(f"\n{'='*60}")
print("완료!")
