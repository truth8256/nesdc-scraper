"""
compare_parsers_v2.py
=====================
개선된 파서 비교 스크립트

기능:
- 체크포인트 기반 재개 (resume)
- 개별 파서 선택 실행
- 진행 상황 저장
- 실행 시간 측정

Usage:
    # 전체 실행
    python compare_parsers_v2.py

    # pdfplumber만 실행
    python compare_parsers_v2.py --parser pdfplumber

    # 재개
    python compare_parsers_v2.py --resume

    # 특정 폴더만
    python compare_parsers_v2.py --folders 15308 15309 15310
"""

import sys
import io
import json
import subprocess
import os
import time
import argparse
from pathlib import Path
from find_target_pages_v2 import find_target_pages_v2

CHECKPOINT_FILE = "parser_comparison_checkpoint.json"
RESULTS_FILE = "parser_comparison_results.json"

def load_checkpoint():
    """체크포인트 파일 로드"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'completed': [], 'results': []}

def save_checkpoint(checkpoint):
    """체크포인트 저장"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def run_parser(parser_name, folder, pages, timeout=120):
    """
    파서 실행

    Returns:
        (success, output_file, error_msg, elapsed_time, has_data)
    """
    if parser_name == 'pdfplumber':
        script = 'pdfplumber_table_parser.py'
        output_file = f'data/parsed_tables/{folder}_pdfplumber.json'
    elif parser_name == 'docling':
        script = 'table_parser.py'
        output_file = f'data/parsed_tables/{folder}_docling.json'
    else:
        return False, None, "Unknown parser", 0, False

    pages_str = ' '.join(map(str, pages))
    cmd = f'py {script} --folder {folder} --pages {pages_str}'

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            if os.path.exists(output_file):
                # Check if file has data
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    has_data = isinstance(data, list) and len(data) > 0

                return True, output_file, None, elapsed, has_data
            else:
                return False, None, "Output file not created", elapsed, False
        else:
            return False, None, result.stderr[:200], elapsed, False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, None, f"Timeout ({timeout}s)", elapsed, False
    except Exception as e:
        elapsed = time.time() - start_time
        return False, None, str(e)[:200], elapsed, False

def get_table_count(output_file):
    """JSON 파일의 테이블 개수 확인"""
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return len(data)
    except:
        pass
    return 0

def compare_parsers(parsers=None, resume=False, folder_list=None):
    """
    파서 비교 실행

    Parameters
    ----------
    parsers : list, optional
        실행할 파서 목록 (예: ['pdfplumber'], ['docling'], 또는 None=둘다)
    resume : bool
        체크포인트에서 재개 여부
    folder_list : list, optional
        특정 폴더만 처리 (예: [15308, 15309])
    """
    if parsers is None:
        parsers = ['pdfplumber', 'docling']

    print("=" * 100)
    print("🔍 파서 비교 테스트 v2")
    print("=" * 100)
    print(f"실행 파서: {', '.join(parsers)}")
    print(f"재개 모드: {'예' if resume else '아니오'}")
    print()

    # 체크포인트 로드
    checkpoint = load_checkpoint() if resume else {'completed': [], 'results': []}

    # 페이지 추천 가져오기
    print("📋 페이지 추천 로딩 (find_target_pages_v2)...")
    recommendations = find_target_pages_v2()

    if not recommendations:
        print("❌ 추천 페이지를 찾을 수 없습니다.")
        return

    # 폴더 필터링
    if folder_list:
        recommendations = {k: v for k, v in recommendations.items() if k in folder_list}
        print(f"필터링: {len(recommendations)}개 폴더 선택됨")

    total_folders = len(recommendations)
    completed_count = 0

    for folder_id, info in sorted(recommendations.items()):
        # 체크포인트 확인 (재개 모드)
        if resume:
            # 이미 완료된 폴더는 건너뛰기
            already_done = [c for c in checkpoint['completed'] if c['folder_id'] == folder_id]
            if already_done:
                # 이 폴더의 모든 파서가 완료되었는지 확인
                done_parsers = set(already_done[0].get('parsers', []))
                remaining_parsers = [p for p in parsers if p not in done_parsers]

                if not remaining_parsers:
                    print(f"⏭️  {folder_id}: 이미 완료됨 (건너뜀)")
                    completed_count += 1
                    continue
                else:
                    print(f"🔄 {folder_id}: 부분 완료 - {remaining_parsers} 실행")
                    parsers_to_run = remaining_parsers
            else:
                parsers_to_run = parsers
        else:
            parsers_to_run = parsers

        print(f"\n{'=' * 100}")
        print(f"📂 {folder_id}: {info['pdf_name']}")
        print(f"   페이지: {info['recommended']}")
        print(f"   진행: {completed_count + 1}/{total_folders}")
        print('=' * 100)

        folder_results = {
            'folder_id': folder_id,
            'pdf_name': info['pdf_name'],
            'pages': info['recommended']
        }

        # 각 파서 실행
        for parser in parsers_to_run:
            print(f"{'1️⃣' if parser == 'pdfplumber' else '2️⃣'} {parser} 실행 중...")

            success, output_file, error, elapsed, has_data = run_parser(
                parser,
                folder_id,
                info['recommended']
            )

            if success and has_data:
                table_count = get_table_count(output_file)
                print(f"   ✅ 성공 - {table_count}개 테이블 - {elapsed:.1f}초")
            elif success and not has_data:
                print(f"   ⚠️  성공했으나 데이터 없음 - {elapsed:.1f}초")
            else:
                print(f"   ❌ 실패 - {error} - {elapsed:.1f}초")

            folder_results[parser] = {
                'success': success and has_data,
                'has_data': has_data,
                'error': error,
                'time': elapsed,
                'table_count': get_table_count(output_file) if output_file else 0
            }

        # 결과 저장
        checkpoint['results'].append(folder_results)
        checkpoint['completed'].append({
            'folder_id': folder_id,
            'parsers': parsers_to_run
        })
        save_checkpoint(checkpoint)

        completed_count += 1

    # 최종 요약
    print_summary(checkpoint['results'], parsers)

    # 최종 결과 파일 저장
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint['results'], f, ensure_ascii=False, indent=2)

    print(f"\n💾 결과 저장: {RESULTS_FILE}")
    print(f"💾 체크포인트: {CHECKPOINT_FILE}")

def print_summary(results, parsers):
    """결과 요약 출력"""
    print("\n" + "=" * 100)
    print("📊 최종 비교 결과")
    print("=" * 100)
    print()

    # 파서별 통계
    stats = {}
    for parser in parsers:
        parser_results = [r[parser] for r in results if parser in r]
        success_count = sum(1 for r in parser_results if r['success'])
        total_time = sum(r['time'] for r in parser_results)
        avg_time = total_time / len(parser_results) if parser_results else 0
        total_tables = sum(r['table_count'] for r in parser_results)

        stats[parser] = {
            'success': success_count,
            'total': len(parser_results),
            'avg_time': avg_time,
            'total_tables': total_tables
        }

    # 테이블 출력
    print(f"{'항목':<20} ", end="")
    for parser in parsers:
        print(f"{parser:>15} ", end="")
    print()
    print("-" * 100)

    print(f"{'성공률':<20} ", end="")
    for parser in parsers:
        s = stats[parser]
        print(f"{s['success']}/{s['total']} ({s['success']/s['total']*100:.1f}%){' '*3} ", end="")
    print()

    print(f"{'평균 실행 시간':<20} ", end="")
    for parser in parsers:
        print(f"{stats[parser]['avg_time']:>14.1f}초 ", end="")
    print()

    print(f"{'총 테이블 수':<20} ", end="")
    for parser in parsers:
        print(f"{stats[parser]['total_tables']:>15}개 ", end="")
    print()

    print()

    # 폴더별 상세
    print("\n" + "=" * 100)
    print("📋 폴더별 상세 결과")
    print("=" * 100)
    print()

    header = f"{'폴더':<8} "
    for parser in parsers:
        header += f"{parser:>12} {'시간':>8} "
    print(header)
    print("-" * 100)

    for r in results:
        row = f"{r['folder_id']:<8} "
        for parser in parsers:
            if parser in r:
                info = r[parser]
                if info['success']:
                    status = f"{info['table_count']}개"
                elif info['has_data']:
                    status = "데이터없음"
                else:
                    status = "실패"
                row += f"{status:>12} {info['time']:>7.1f}s "
            else:
                row += f"{'N/A':>12} {'':>8} "
        print(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="파서 비교 v2")
    parser.add_argument(
        '--parser',
        choices=['pdfplumber', 'docling', 'both'],
        default='both',
        help="실행할 파서 선택"
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="체크포인트에서 재개"
    )
    parser.add_argument(
        '--folders',
        nargs='+',
        type=int,
        help="특정 폴더만 처리 (예: --folders 15308 15309)"
    )

    args = parser.parse_args()

    # 파서 목록
    if args.parser == 'both':
        parsers = ['pdfplumber', 'docling']
    else:
        parsers = [args.parser]

    compare_parsers(
        parsers=parsers,
        resume=args.resume,
        folder_list=args.folders
    )
