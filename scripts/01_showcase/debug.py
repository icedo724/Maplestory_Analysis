import pandas as pd
import numpy as np
import os

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(BASE_DIR, "data", "showcase")
PROCESSED_DIR = os.path.join(SAVE_DIR, "preprocessed")
FILE_PATH = os.path.join(PROCESSED_DIR, "daily_segment_processed.csv")

# 진단 임계치 설정
ZERO_RATIO_THRESHOLD = 0.5  # 유저의 50% 이상이 0이면 이상치로 간주
SPIKE_THRESHOLD = 5.0  # 평소보다 5배 이상 높은 경험치 획득 시 이상치로 간주


# ==========================================

def run_logic_check():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {FILE_PATH}")
        return

    df = pd.read_csv(FILE_PATH)
    daily_cols = [c for c in df.columns if c.startswith('Daily_')]

    report = []

    print(f"🔍 총 {len(daily_cols)}일치 데이터 논리 검사 시작...")
    print("-" * 60)

    for col in daily_cols:
        date_str = col.replace('Daily_', '')
        data = df[col].dropna()

        if data.empty:
            report.append({"날짜": date_str, "이슈": "데이터 공백", "내용": "해당 날짜 데이터가 비어있음"})
            continue

        # 1. 수집 정체(Zero Inflation) 검사
        zero_ratio = (data == 0).mean()

        # 2. 비정상 폭등(Spike) 검사
        # 다른 날짜들의 평균 대비 해당 날짜의 평균을 비교
        other_cols = [c for c in daily_cols if c != col]
        global_mean = df[other_cols].mean().mean()
        current_mean = data.mean()

        # 논리적 이상 징후 판단
        if zero_ratio >= ZERO_RATIO_THRESHOLD:
            report.append({
                "날짜": date_str,
                "이슈": "수집 정체/누락",
                "내용": f"유저의 {zero_ratio * 100:.1f}%가 경험치 변화 없음 (API 미갱신 의심)"
            })

        if current_mean > global_mean * SPIKE_THRESHOLD:
            report.append({
                "날짜": date_str,
                "이슈": "데이터 폭등(Spike)",
                "내용": f"평균 대비 {current_mean / global_mean:.1f}배 높은 수치 감지 (데이터 중복 합산 의심)"
            })

    # 결과 출력
    if report:
        print(f"🚨 총 {len(report)}건의 논리적 이상 일자가 발견되었습니다.")
        print("-" * 60)
        report_df = pd.DataFrame(report)
        print(report_df.to_string(index=False))
        print("-" * 60)
        print("💡 위 날짜들은 대시보드 분석 시 '이상치'로 작용하여 통계 결과(P-value)를 왜곡할 수 있습니다.")
    else:
        print("✅ 모든 데이터가 논리적으로 정상 범위 내에 있습니다.")


if __name__ == "__main__":
    run_logic_check()