"""
test_custom_table_parser.py
============================
custom_table_parser.py의 CLI 테스트 진입점.

사용법:
  conda run -n paddle311 python test_custom_table_parser.py --page 12
  conda run -n paddle311 python test_custom_table_parser.py --page 12 --debug-image
  conda run -n paddle311 python test_custom_table_parser.py --image data/page12_sample.png
"""

import os
import sys
import argparse

# PaddleOCR 환경변수 (import 전 설정)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"

# 현재 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_table_parser import parse_table

DEFAULT_PDF = (
    r"D:\working\polldata\nesdc_scraper\data\raw\15380"
    r"\3.통계표(260213)_MBC_2026년+설날특집+정치·사회현안+여론조사(2차)(15일보도).pdf"
)
DEFAULT_OUTPUT_DIR = r"D:\working\polldata\nesdc_scraper\data\custom_output"


def main():
    parser = argparse.ArgumentParser(
        description="투영 프로파일 기반 자체 표 인식기 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
파라미터 튜닝 가이드:
  --row-min-gap    : 행 분리 최소 갭 (기본 3). 작게 → 더 많은 행 탐지
  --col-min-gap    : 열 분리 최소 갭 (기본 10). 크게 → 더 적은 열 합산
  --binarize-thr   : 이진화 임계값 (기본 200). 낮추면 더 민감
  --debug-image    : 행/열 탐지 시각화 이미지 저장

예시:
  conda run -n paddle311 python test_custom_table_parser.py --page 12 --debug-image
  conda run -n paddle311 python test_custom_table_parser.py --page 12 --col-min-gap 20
        """
    )
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF, help="PDF 파일 경로")
    parser.add_argument("--image", type=str, default=None, help="이미지 직접 지정 (PDF 대신)")
    parser.add_argument("--page", type=int, default=12, help="파싱할 페이지 (기본: 12)")
    parser.add_argument("--dpi", type=int, default=200, help="PDF→이미지 DPI (기본: 200)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="결과 저장 경로")

    # 이진화
    parser.add_argument("--binarize-thr", type=int, default=200, help="이진화 임계값 (기본: 200)")

    # 행 탐지
    parser.add_argument("--row-min-gap", type=int, default=3, help="행 분리 최소 갭 픽셀 (기본: 3)")
    parser.add_argument("--row-h-threshold", type=float, default=0.02, help="H-투영 임계값 비율 (기본: 0.02)")
    parser.add_argument("--row-min-height", type=int, default=5, help="최소 행 높이 픽셀 (기본: 5)")

    # 열 탐지
    parser.add_argument("--col-min-gap", type=int, default=10, help="열 분리 최소 갭 픽셀 (기본: 10)")
    parser.add_argument("--col-v-threshold", type=float, default=0.01, help="V-투영 임계값 비율 (기본: 0.01)")

    # OCR
    parser.add_argument("--cell-pad", type=int, default=4, help="셀 크롭 패딩 픽셀 (기본: 4)")

    # 출력
    parser.add_argument("--debug-image", action="store_true", help="행/열 탐지 시각화 이미지 저장")
    parser.add_argument("--debug-scale", type=float, default=0.5, help="시각화 이미지 스케일 (기본: 0.5)")

    args = parser.parse_args()

    # PDF 파일 존재 확인
    if not args.image and not os.path.isfile(args.pdf):
        print(f"[오류] PDF 파일을 찾을 수 없습니다: {args.pdf}")
        sys.exit(1)
    if args.image and not os.path.isfile(args.image):
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {args.image}")
        sys.exit(1)

    result = parse_table(
        pdf_path=args.pdf,
        page=args.page,
        dpi=args.dpi,
        output_dir=args.output_dir,
        binarize_threshold=args.binarize_thr,
        row_min_gap=args.row_min_gap,
        row_h_threshold=args.row_h_threshold,
        row_min_height=args.row_min_height,
        col_min_gap=args.col_min_gap,
        col_v_threshold=args.col_v_threshold,
        cell_pad=args.cell_pad,
        debug_image=args.debug_image,
        debug_scale=args.debug_scale,
        image_path=args.image,
    )

    if result:
        df = result["dataframe"]
        print(f"\n{'='*60}")
        print(f"[최종 결과]")
        print(f"  행 밴드 수   : {len(result['row_bands'])}")
        print(f"  열 밴드 수   : {len(result['col_bands'])}")
        print(f"  DataFrame    : {len(df)} 행 × {len(df.columns)} 열")
        print(f"  CSV 저장     : {result['csv_path']}")
        print(f"  JSON 저장    : {result['json_path']}")
        print(f"\n[DataFrame 전체]\n")
        print(df.to_string())


if __name__ == "__main__":
    main()
