"""
schema.py
=========
NESDC 여론조사 파싱 결과의 표준 JSON 스키마 정의 및 변환 유틸.

표준 스키마 (SurveyTable):
{
    "poll_id":    str,           # 여심위 nttId
    "page":       int | list,    # 원본 PDF 페이지 번호
    "keyword":    str,           # 수집 명령 키워드 ("정당지지도" 등)
    "question":   str,           # 표 위 문항 텍스트 전문
    "method":     str,           # 파서명+버전 ("pdfplumber_v1.0")
    "columns":    [str, ...],    # 응답 선택지 목록 (모든 행 동일)
    "data": [
        {
            "group":      str,   # 인구통계 그룹 ("전체","성별","연령",...)
            "category":   str,   # 세부 항목 ("남성","30대","서울",...)
            "n":          int?,  # 사례수 (단일)
            "n_raw":      int?,  # 조사완료 사례수
            "n_weighted": int?,  # 가중 사례수
            "responses":  {str: float}  # 응답항목: 비율(%)
        }, ...
    ]
}
"""

import re
from typing import Any

# ─────────────────────────────────────────────
# 1. 그룹 자동 추론
# ─────────────────────────────────────────────

# (패턴, 그룹명) 순서대로 매칭
_GROUP_RULES: list[tuple[list[str], str]] = [
    # 전체
    (["전체", "전 체", "합계", "Total", "■전체■", "■ 전 체 ■"], "전체"),
    # 성별
    (["남성", "여성", "남자", "여자"], "성별"),
    # 연령
    ([
        "18-29", "18~29", "만 18", "19-29", "20대", "30대", "40대", "50대",
        "60대", "70대", "60세", "70세", "세이상",
    ], "연령"),
    # 연령+성별 교차
    ([
        "남성$", "여성$",  # 뒤에 성별이 붙는 패턴 (30대남성 등)
    ], "연령성별"),
    # 권역/지역
    ([
        "서울", "인천", "경기", "대전", "세종", "충남", "충북", "충청",
        "광주", "전남", "전북", "전라", "대구", "경북", "경상",
        "부산", "울산", "경남", "강원", "제주",
    ], "권역"),
    # 직업
    ([
        "사무", "관리", "전문직", "판매", "생산", "노무", "서비스",
        "자영업", "가정주부", "주부", "학생", "무직", "은퇴", "농업",
        "임업", "축산", "어업",
    ], "직업"),
    # 정치이념
    (["보수", "중도", "진보"], "정치이념"),
    # 지지정당
    ([
        "더불어", "민주당", "국민의힘", "조국혁신", "개혁신당",
        "진보당", "새로운미래", "무당층", "지지정당없음",
    ], "지지정당"),
    # 대통령 국정운영 평가
    ([
        "매우잘한다", "잘하는편", "잘못하는편", "매우잘못",
        "잘모르겠다",
    ], "국정운영평가"),
    # 학력
    (["중졸", "고졸", "대졸", "대학원", "대재"], "학력"),
    # 소득
    (["만원", "소득"], "소득"),
]


def infer_group(category: str) -> str:
    """
    카테고리 문자열에서 인구통계 그룹을 자동 추론.

    Examples:
        infer_group("남성")   -> "성별"
        infer_group("30대")   -> "연령"
        infer_group("서울")   -> "권역"
        infer_group("전체")   -> "전체"
        infer_group("30대남성") -> "연령성별"
    """
    cat = category.replace(" ", "").strip()

    if not cat:
        return "미분류"

    # "전체" 를 먼저 체크
    if cat in ("전체", "전 체", "합계", "Total"):
        return "전체"

    # 연령+성별 교차 패턴 (예: "30대남성", "만 18-29세여성")
    age_pattern = re.search(
        r'(\d{2}[~\-]?\d{0,2}세?|[234567]0대|세이상)', cat
    )
    gender_pattern = re.search(r'(남성|여성|남자|여자)$', cat)
    if age_pattern and gender_pattern:
        return "연령성별"

    # 일반 규칙 매칭
    for keywords, group_name in _GROUP_RULES:
        if group_name in ("전체", "연령성별"):
            continue  # 이미 위에서 처리
        for kw in keywords:
            if kw in cat:
                return group_name

    return "기타"


# ─────────────────────────────────────────────
# 2. 사례수 파싱 유틸
# ─────────────────────────────────────────────

def _parse_n(value) -> int | None:
    """사례수 문자열 → 정수 변환. '(1,076)' → 1076"""
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r'[(),\s]', '', s)
    if s in ('', '-', '.'):
        return None
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────
# 3. Flat 테이블 → 표준 스키마 변환
# ─────────────────────────────────────────────

def _detect_column_roles(columns: list[str]) -> dict:
    """
    컬럼 목록을 분류: header, base, value 로 나눈다.
    Returns dict with keys: 'header_cols', 'base_cols', 'value_cols'
    """
    header_cols = []
    base_cols = []
    value_cols = []

    base_keywords = ['사례수', 'Base', 'base', '가중값', '가중사례수',
                     '조사완료', 'n_raw', 'n_weighted', '가중']

    for i, col in enumerate(columns):
        col_str = str(col).strip()

        # 빈 컬럼명 또는 첫 번째 컬럼 → 보통 구분/헤더
        if col_str == '' or col_str == '구분':
            header_cols.append(col)
        elif any(bk in col_str for bk in base_keywords):
            base_cols.append(col)
        elif i == 0:
            header_cols.append(col)
        else:
            value_cols.append(col)

    return {
        'header_cols': header_cols,
        'base_cols': base_cols,
        'value_cols': value_cols,
    }


def _clean_column_name(col_name: str) -> str:
    """
    Docling이 컬럼명에 붙인 값을 제거.
    '더불어 민주당.45.6' → '더불어 민주당'
    'Base= 전체 ( 단위 : %).■전체■' → '구분'
    """
    s = str(col_name).strip()

    # 패턴: "이름.숫자" → "이름"
    m = re.match(r'^(.+?)\.\d+[\.\d]*$', s)
    if m:
        return m.group(1).strip()

    # "Base= 전체..." 류 → 구분
    if s.startswith('Base=') or '■전체■' in s or '■ 전 체 ■' in s:
        return '구분'

    return s


def _clean_value(val) -> float | None:
    """값 문자열 → float 변환. '-' → 0.0, 빈값 → None"""
    if val is None:
        return None
    s = str(val).strip()
    if s in ('-', '.', ''):
        return 0.0
    s = re.sub(r'[%(),\s]', '', s)
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def convert_flat_to_standard(
    flat_data: list[dict],
    *,
    poll_id: str = "",
    page: int | list[int] = 0,
    keyword: str = "",
    question: str = "",
    method: str = "unknown_v0",
) -> dict[str, Any]:
    """
    기존 flat 테이블 (row-oriented dict 리스트) → 표준 스키마로 변환.

    Parameters
    ----------
    flat_data : list[dict]
        기존 파서 출력의 data 배열.
        예: [{"구분":"남성", "사례수":577, "더불어민주당":45.6, ...}, ...]
    poll_id, page, keyword, question, method : 메타 정보

    Returns
    -------
    dict  표준 SurveyTable 딕셔너리
    """
    if not flat_data:
        return {
            "poll_id": poll_id, "page": page, "keyword": keyword,
            "question": question, "method": method,
            "columns": [], "data": [],
        }

    # 컬럼 추출
    raw_columns = list(flat_data[0].keys())
    roles = _detect_column_roles(raw_columns)

    # 깨끗한 응답 컬럼명 목록
    clean_value_cols = [_clean_column_name(c) for c in roles['value_cols']]
    # 중복 제거된 컬럼 목록 (순서 유지)
    columns = list(dict.fromkeys(clean_value_cols))

    # 현재 그룹 추적 (빈 셀이면 이전 그룹 유지)
    current_group = ""

    standard_data = []
    for row in flat_data:
        # 카테고리 추출 (header 컬럼들에서)
        category = ""
        for hc in roles['header_cols']:
            v = str(row.get(hc, "")).strip()
            if v and v not in ('', 'nan'):
                # 그룹 헤더인 경우 (성별, 연령별 등)
                if _is_group_header(v):
                    current_group = v.replace("별", "")
                    continue
                category = v
                break

        if not category:
            continue

        # 그룹 추론
        group = current_group if current_group else infer_group(category)
        if not current_group:
            current_group = group

        # 사례수 추출
        n_raw = None
        n_weighted = None
        n = None

        for bc in roles['base_cols']:
            bc_str = str(bc)
            val = _parse_n(row.get(bc))
            if '가중' in bc_str:
                n_weighted = val
            elif '조사완료' in bc_str or '사례수' in bc_str:
                if n_raw is None:
                    n_raw = val
                else:
                    n_weighted = val
            else:
                if n is None:
                    n = val

        if n is None and n_raw is None:
            # 단일 사례수
            for bc in roles['base_cols']:
                val = _parse_n(row.get(bc))
                if val is not None:
                    n = val
                    break

        # 응답값 추출
        responses = {}
        for raw_col, clean_col in zip(roles['value_cols'], clean_value_cols):
            val = _clean_value(row.get(raw_col))
            if val is not None:
                responses[clean_col] = val

        entry = {
            "group": group,
            "category": category.replace(" ", "").strip(),
        }

        # n 필드: 둘 다 있으면 n_raw + n_weighted, 하나만 있으면 n
        if n_raw is not None or n_weighted is not None:
            if n_raw is not None:
                entry["n_raw"] = n_raw
            if n_weighted is not None:
                entry["n_weighted"] = n_weighted
        elif n is not None:
            entry["n"] = n

        entry["responses"] = responses
        standard_data.append(entry)

    return {
        "poll_id": poll_id,
        "page": page,
        "keyword": keyword,
        "question": question,
        "method": method,
        "columns": columns,
        "data": standard_data,
    }


def _is_group_header(value: str) -> bool:
    """성별, 연령별 등의 그룹 헤더 행인지 판별"""
    keywords = [
        '성별', '연령', '지역', '권역', '직업', '학력', '소득',
        '이념', '지지정당', '기초', '광역', '연령별', '거주지역',
        '연령성별', '정치 이 념 성향', '정치이념성향', '대통령',
        '국정운영', '평가',
    ]
    v = value.replace(" ", "").strip()
    return any(k.replace(" ", "") in v for k in keywords)


# ─────────────────────────────────────────────
# 4. 스키마 검증
# ─────────────────────────────────────────────

def validate_schema(table: dict) -> list[str]:
    """
    표준 스키마 준수 여부를 검증.
    Returns: 문제 목록 (빈 리스트 = 통과)
    """
    issues = []

    required_fields = ["poll_id", "page", "method", "columns", "data"]
    for f in required_fields:
        if f not in table:
            issues.append(f"Missing required field: '{f}'")

    if "data" not in table:
        return issues

    data = table["data"]
    columns = set(table.get("columns", []))

    if not data:
        issues.append("Empty data array")
        return issues

    for i, entry in enumerate(data):
        # 필수 필드
        if "group" not in entry:
            issues.append(f"data[{i}]: missing 'group'")
        if "category" not in entry:
            issues.append(f"data[{i}]: missing 'category'")
        if "responses" not in entry:
            issues.append(f"data[{i}]: missing 'responses'")
            continue

        # 사례수 확인
        has_n = any(k in entry for k in ("n", "n_raw", "n_weighted"))
        if not has_n:
            issues.append(f"data[{i}] ({entry.get('category','')}): no sample size (n)")

        # responses 키가 columns와 일치하는지
        resp_keys = set(entry["responses"].keys())
        if columns and resp_keys != columns:
            missing = columns - resp_keys
            extra = resp_keys - columns
            if missing:
                issues.append(
                    f"data[{i}] ({entry.get('category','')}): "
                    f"missing response keys: {missing}"
                )
            if extra:
                issues.append(
                    f"data[{i}] ({entry.get('category','')}): "
                    f"extra response keys: {extra}"
                )

    return issues


# ─────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 간단한 테스트
    print("=== infer_group 테스트 ===")
    test_cases = [
        ("전체", "전체"),
        ("남성", "성별"),
        ("여성", "성별"),
        ("30대", "연령"),
        ("만 18-29세", "연령"),
        ("60세 이상", "연령"),
        ("30대남성", "연령성별"),
        ("만 18-29세여성", "연령성별"),
        ("서울", "권역"),
        ("인천", "권역"),
        ("경기 . 인천", "권역"),
        ("사무 / 관리 / 전문직", "직업"),
        ("가정주부", "직업"),
        ("보수", "정치이념"),
        ("중도", "정치이념"),
        ("더불어민주당", "지지정당"),
        ("알 수 없음", "기타"),
    ]

    passed = 0
    for category, expected in test_cases:
        result = infer_group(category)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} infer_group('{category}') = '{result}' (expected: '{expected}')")

    print(f"\n결과: {passed}/{len(test_cases)} 통과")
