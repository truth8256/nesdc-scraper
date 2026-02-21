# 멀티 컴퓨터 환경 설정 가이드

## 🖥️ 새 컴퓨터에서 시작하기

### 1. 저장소 클론
```bash
git clone <repository-url>
cd nesdc-scraper
```

### 2. Python 가상환경 설정
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 데이터 폴더 구조 생성
```bash
mkdir -p data/raw
mkdir -p data/parsed_tables
```

### 5. (선택) 환경 변수 설정
필요한 경우 `.env` 파일 생성:
```
GEMINI_API_KEY=your_api_key_here
HF_HOME=path/to/huggingface/cache
```

---

## ⚠️ 주의사항

### Git에 포함되지 않는 파일들
다음 파일/폴더는 `.gitignore`에 포함되어 **자동 동기화되지 않습니다**:

- `data/raw/` - PDF 원본 파일 (수동 복사 필요)
- `data/parsed_tables/` - 파싱 결과 (재생성 가능)
- `page_index.csv` - 페이지 인덱스 (재생성 가능)
- `*_checkpoint.json` - 체크포인트 파일 (컴퓨터별)
- `.venv/` - 가상환경 (컴퓨터별로 새로 생성)

### PDF 파일 동기화 방법

**옵션 1**: 수동 복사
```bash
# USB나 클라우드에서 복사
cp -r /path/to/backup/data/raw/* data/raw/
```

**옵션 2**: Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "data/raw/*.pdf"
git add .gitattributes
```

**옵션 3**: 클라우드 동기화
- OneDrive, Google Drive, Dropbox 등에 `data/raw/` 폴더 동기화

---

## 🔄 작업 전 체크리스트

### 컴퓨터 A에서 작업 완료 시
```bash
# 1. 코드 변경사항 커밋
git add .
git commit -m "작업 내용 설명"
git push

# 2. (선택) 중요한 생성 파일 백업
# page_index.csv, checkpoint 파일 등
```

### 컴퓨터 B에서 작업 시작 시
```bash
# 1. 최신 코드 가져오기
git pull

# 2. 패키지 업데이트 (필요시)
pip install -r requirements.txt

# 3. 데이터 파일 확인
ls data/raw/  # PDF 파일이 있는지 확인
```

---

## 🐛 문제 해결

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: data/raw/..."
- PDF 파일을 `data/raw/` 폴더에 복사

### Docling 모델 로딩 오류
```bash
# 모델 캐시 재다운로드
rm -rf ~/.cache/huggingface/hub/models--ds4sd--docling*
python -c "from transformers import snapshot_download; snapshot_download('ds4sd/docling-models')"
```

### Git 충돌 발생
```bash
# 현재 작업 임시 저장
git stash

# 최신 코드 가져오기
git pull

# 임시 저장한 작업 복원
git stash pop

# 충돌 해결 후
git add .
git commit -m "충돌 해결"
```

---

## 📝 권장 워크플로우

### 병렬 작업 피하기
- 같은 파일을 동시에 수정하지 않기
- 작업 시작 전 항상 `git pull`
- 작업 완료 후 즉시 `git push`

### 브랜치 사용 (권장)
```bash
# 컴퓨터 A
git checkout -b feature/improve-parser
# ... 작업 ...
git push -u origin feature/improve-parser

# 컴퓨터 B
git checkout -b feature/add-validation
# ... 작업 ...
git push -u origin feature/add-validation

# 완료 후 main에 병합
git checkout main
git merge feature/improve-parser
```

---

## 🎯 핵심 요약

| 항목 | 동기화 방법 |
|------|-------------|
| **코드 (.py)** | ✅ Git |
| **설정 파일** | ✅ Git |
| **문서 (.md)** | ✅ Git |
| **PDF 원본** | ⚠️ 수동 복사 또는 클라우드 |
| **파싱 결과** | ❌ 재생성 (git 제외) |
| **가상환경** | ❌ 각 컴퓨터에서 생성 |
| **체크포인트** | ❌ 컴퓨터별 독립적 |

---

생성일: 2026-02-21
