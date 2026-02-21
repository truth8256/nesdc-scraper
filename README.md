
# NESDC 여론조사 데이터 수집 및 분석 시스템

중앙선거여론조사심의위원회(NESDC)의 여론조사 결과를 **수집(Scraping)** → **표 추출(Parsing)** → **검증(Validation)**하는 파이프라인입니다.

## 📂 프로젝트 구조

```
nesdc_scraper/
├── scraper.py                    # [수집] 핵심 스크래핑 로직 (Playwright)
├── run_daily_update.py           # [수집] 일일 업데이트 실행기
├── run_full_crawl.py             # [수집] 전체 데이터 크롤링 실행기
├── daily_update.bat              # [자동화] 윈도우 작업 스케줄러용 배치 파일
│
├── table_parser.py               # [파싱] Docling 기반 표 추출 + LLM 폴백
├── custom_table_parser.py        # [파싱] 투영 프로파일 + OCR 기반 자체 표 파서
├── pdfplumber_table_parser.py    # [파싱] pdfplumber 기반 텍스트 레이어 표 파서
├── llm_extractor.py              # [파싱] LLM(OpenAI/Gemini) 기반 표 추출기
├── validator.py                  # [검증] 추출된 표 데이터 유효성 검사
│
├── test_custom_table_parser.py   # [테스트] custom_table_parser CLI 테스트
├── test_paddle_table_parser.py   # [테스트] PaddleOCR 기반 표 파서 테스트
├── test_pdfplumber.py            # [테스트] pdfplumber 직접 테스트
├── test_upstage_table_parser.py  # [테스트] Upstage API 기반 표 파서 테스트
│
├── requirements.txt              # 의존성 목록
├── research_ai_table_extraction.md  # AI 서비스 비교 분석 메모
│
├── tools/                        # 보조 유틸리티 스크립트
│   ├── analyze_pdfs.py           #   PDF 구조 분석
│   ├── extract_survey.py         #   서베이 데이터 추출
│   ├── save_tables_csv.py        #   표 → CSV 저장
│   ├── save_tables_md.py         #   표 → Markdown 저장
│   ├── view_tables.py            #   표 데이터 조회
│   ├── test_docling.py           #   Docling 테스트
│   └── check_docling_args.py     #   Docling 인자 확인
│
├── data/
│   ├── metadata/                 # 수집된 메타데이터 CSV 및 상태 파일
│   ├── raw/                      # PDF 및 원본 파일 저장소
│   └── parsed_tables/            # 파싱된 JSON 데이터 저장소
│
└── .github/workflows/
    └── daily_update.yml          # GitHub Actions 자동 수집 워크플로우
```

## 🛠️ 주요 구성 요소

### 1. 데이터 수집기 (Scraper)
- **`scraper.py`**: NESDC 웹사이트에서 여론조사 정보를 수집하고 PDF 파일을 다운로드합니다. `Playwright`를 사용하여 동적 웹페이지를 탐색합니다.
- **`run_daily_update.py`**: 매일 실행하여 최신 게시글을 수집합니다. 엠바고로 미수집된 건도 재확인합니다. → `data/metadata/polls_metadata_daily_clean.csv`
- **`run_full_crawl.py`**: 과거 전체 데이터를 수집합니다. 중단 시점부터 이어하기(Resume) 가능. → `data/metadata/polls_metadata_full.csv`

### 2. 표 파서 (Table Parser)
PDF에서 설문 조사 표를 추출하는 여러 전략이 구현되어 있습니다.

| 파서 | 방식 | 특징 |
|---|---|---|
| `table_parser.py` | Docling → Markdown → DataFrame | 메인 파서. LLM 폴백 지원 |
| `custom_table_parser.py` | 투영 프로파일 + RapidOCR | 행/열 간격이 좁은 난이도 높은 표 대응 |
| `pdfplumber_table_parser.py` | pdfplumber 텍스트 레이어 | OCR 불필요. 텍스트 레이어 내장 PDF 전용 |
| `llm_extractor.py` | OpenAI GPT-4o-mini / Gemini | Docling 실패 시 LLM으로 표 구조 추출 |

### 3. 데이터 검증기 (Validator)
- **`validator.py`**: 파싱된 JSON의 품질을 자동 평가합니다.
  - **행 합계 검증**: 응답 비율 합이 100%(±2%) 이내인지 확인
  - **소계(Subtotal) 처리**: 부분합 알고리즘으로 소계 열 포함 표도 정확히 검증
  - 결과 분류: `Fully Valid` / `Mostly Valid` / `Collection Impossible`

## 🚀 사용 예시

### 일일 수집 실행
```bash
python run_daily_update.py
```

### 특정 폴더 파싱 및 검증
```bash
# 파싱 (PDF → JSON, 특정 페이지 지정)
python table_parser.py --folder 15334 --pages 10 13

# 검증 (JSON 확인)
python validator.py --input data/parsed_tables/15334_tables.json
```

### 자체 표 파서 실행
```bash
# 투영 프로파일 + OCR 기반 파싱 (conda 환경)
conda run -n paddle311 python test_custom_table_parser.py --page 12
conda run -n paddle311 python test_custom_table_parser.py --page 12 --debug-image
```
