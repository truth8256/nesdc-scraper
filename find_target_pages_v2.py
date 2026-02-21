"""
find_target_pages_v2.py
=======================
개선된 페이지 탐지 로직 (사례수/조사완료 조건 추가)

개선 사항:
1. 키워드 "정당" 매칭 (기존)
2. 페이지 3 이상 (수정)
3. 표 데이터 우선 (기존)
4. [NEW] 표 상단에 "사례수", "조사완료", "가중값" 포함 필수
"""

import sys
import io
import pandas as pd

# UTF-8 출력 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

def find_target_pages_v2(csv_path="page_index.csv", keyword="정당", min_page=3):
    """
    개선된 정당지지도 페이지 탐지

    필수 조건:
    1. 키워드 포함
    2. min_page 이상
    3. 표 데이터에 "사례수", "조사완료", "가중값" 포함
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 키워드 검색
    mask_keyword = (
        df['text_head'].str.contains(keyword, na=False, case=False) |
        df['table_head'].str.contains(keyword, na=False, case=False)
    )

    # 페이지 필터
    mask_page = df['page'] >= min_page

    # 표 상단에 "사례수", "조사완료", "가중값" 포함 확인
    mask_table_indicators = (
        df['table_head'].str.contains('사례수', na=False, case=False) |
        df['table_head'].str.contains('조사완료', na=False, case=False) |
        df['table_head'].str.contains('가중값', na=False, case=False)
    )

    # 모든 조건 결합
    mask_all = mask_keyword & mask_page & mask_table_indicators

    result = df[mask_all].copy()

    if len(result) == 0:
        print(f"❌ 조건을 만족하는 페이지를 찾지 못했습니다.")
        print(f"   - 키워드: {keyword}")
        print(f"   - 최소 페이지: {min_page}")
        print(f"   - 표 지표: 사례수, 조사완료, 가중값")
        return {}

    # 표 데이터 품질 확인
    result['has_good_table'] = (
        result['table_head'].notna() &
        (result['table_head'].str.len() > 50)  # 충분한 길이의 표 데이터
    )

    # 폴더별로 분석
    recommendations = {}

    for folder_id in sorted(result['id'].unique()):
        folder_data = result[result['id'] == folder_id].copy()

        # 품질 좋은 표가 있는 페이지 우선
        good_table_pages = folder_data[folder_data['has_good_table']]['page'].tolist()
        all_pages = folder_data['page'].tolist()

        # 추천: 품질 좋은 표 페이지 우선, 없으면 전체에서 선택
        # 제한 없이 모든 조건 만족 페이지 포함 (유효성 검사가 나중에 필터링)
        if good_table_pages:
            recommended = sorted(good_table_pages)
        else:
            recommended = sorted(all_pages)

        recommendations[int(folder_id)] = {
            'pdf_name': folder_data.iloc[0]['pdf_name'],
            'all_pages': sorted(all_pages),
            'good_table_pages': sorted(good_table_pages),
            'recommended': recommended
        }

    return recommendations


def print_summary(recommendations, actual_pages=None):
    """
    결과 요약 출력

    Parameters
    ----------
    recommendations : dict
        탐지된 페이지 정보
    actual_pages : dict, optional
        실제 정당지지도 페이지 (검증용)
    """
    print("=" * 100)
    print("정당지지도 페이지 탐지 결과 (v2 - 개선)")
    print("=" * 100)
    print()

    found_folders = []
    not_found_folders = []

    # 20개 폴더 전체 확인
    for folder_id in range(15300, 15323):
        if folder_id in [15301, 15313, 15318]:  # 누락된 번호
            continue

        if folder_id in recommendations:
            found_folders.append(folder_id)
            info = recommendations[folder_id]

            print(f"✅ {folder_id}: {info['pdf_name']}")
            print(f"   🎯 추천 페이지: {info['recommended']}")

            # 실제 페이지와 비교 (제공된 경우)
            if actual_pages and folder_id in actual_pages:
                actual = actual_pages[folder_id]
                recommended_set = set(info['recommended'])
                actual_set = set(actual)

                match = recommended_set & actual_set
                if match:
                    print(f"   ✓ 정답 일치: {sorted(match)}")
                else:
                    print(f"   ✗ 정답 불일치. 실제: {actual}")

            pages_str = ' '.join(map(str, info['recommended']))
            print(f"   💡 python pdfplumber_table_parser.py --folder {folder_id} --pages {pages_str}")
            print()
        else:
            not_found_folders.append(folder_id)

    # 못 찾은 폴더
    if not_found_folders:
        print("=" * 100)
        print("❌ 정당지지도 페이지를 찾지 못한 폴더:")
        print("=" * 100)
        for folder_id in not_found_folders:
            print(f"   - {folder_id}")
        print()

    # 요약
    print("=" * 100)
    print("📊 요약")
    print("=" * 100)
    print(f"✅ 찾음: {len(found_folders)}/20 폴더")
    print(f"❌ 못찾음: {len(not_found_folders)}/20 폴더")

    # 정확도 계산 (실제 페이지 제공된 경우)
    if actual_pages:
        print()
        print("=" * 100)
        print("🎯 정확도 검증")
        print("=" * 100)

        total_correct = 0
        total_folders = 0

        for folder_id in sorted(recommendations.keys()):
            if folder_id in actual_pages:
                total_folders += 1
                recommended = set(recommendations[folder_id]['recommended'])
                actual = set(actual_pages[folder_id])

                if recommended & actual:  # 교집합이 있으면 성공
                    total_correct += 1

        if total_folders > 0:
            accuracy = (total_correct / total_folders) * 100
            print(f"정확도: {total_correct}/{total_folders} ({accuracy:.1f}%)")
            print(f"(추천 페이지 중 최소 1개 이상이 실제 페이지와 일치)")

    return found_folders, not_found_folders


if __name__ == "__main__":
    # 사용자가 직접 확인한 실제 정당지지도 페이지
    ACTUAL_PAGES = {
        15300: [6],
        15302: [10],
        15303: [8],
        15304: [5, 6],
        15305: [15],
        15308: [4],
        15309: [4],
        15310: [11, 12],
        15311: [19, 20],
        15312: [25, 26],
        15315: [4],
        15317: [5],
        15319: [8],
        15321: [15, 16],
        15322: [15],
    }

    print("🔍 개선된 탐지 로직 (v2)")
    print("   추가 조건: 표 상단에 '사례수', '조사완료', '가중값' 포함")
    print()

    recommendations = find_target_pages_v2()
    found, not_found = print_summary(recommendations, actual_pages=ACTUAL_PAGES)
