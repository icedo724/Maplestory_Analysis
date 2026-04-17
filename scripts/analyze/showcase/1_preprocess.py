import numpy as np
import os
import sys

import pandas as pd

# scripts/analyze/ 를 경로에 추가하여 utils 임포트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_segment, filter_completed_dates, compute_daily_exp

# ================= CONFIG =================
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", "showcase")

INPUT_FILE  = os.path.join(BASE_DIR, "data", "raw", "daily_tracking_lv.csv")
OUTPUT_FILE = os.path.join(PREPROCESSED_DIR, "daily_segment_processed.csv")
LOG_FILE    = os.path.join(BASE_DIR, "data", "raw", "completed_log.txt")

SHOWCASE_DATE      = "2025-12-13"
INACTIVE_THRESHOLD = 14   # 연속 비활성 일수 기준 — 이 값을 초과하면 이탈 유저로 분류
# ==========================================


def preprocess_for_analysis():
    if not os.path.exists(INPUT_FILE):
        print("[오류] 원본 데이터 파일이 없습니다. 수집을 먼저 진행해주세요.")
        sys.exit()

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # ── 1. 원본 로드 ──────────────────────────────────────────────────────────
    print("[진행] 원본 데이터 로드 중...")
    df = pd.read_csv(INPUT_FILE)

    # ── 2. 날짜 컬럼 정렬 + 완료일 필터 ──────────────────────────────────────
    # [개선] 명시적 datetime 파싱 정렬 — 문자열 정렬에 의존하지 않는다.
    exp_cols = sorted(
        [c for c in df.columns if c.startswith('Exp_')],
        key=lambda x: pd.to_datetime(x.replace('Exp_', ''))
    )
    lv_cols_sorted = sorted(
        [c for c in df.columns if c.startswith('Lv_')],
        key=lambda x: pd.to_datetime(x.replace('Lv_', ''))
    )
    dates = [c.replace('Exp_', '') for c in exp_cols]

    # 수집이 완료된 날짜만 사용
    # Daily 컬럼은 "전날 → 당일" 증분이므로 미완료일이 섞이면 이전 날의 diff 도 오염됨.
    dates = filter_completed_dates(dates, LOG_FILE)
    exp_cols       = [f'Exp_{d}' for d in dates]
    lv_cols_sorted = [f'Lv_{d}'  for d in dates]

    if len(dates) < 2:
        print("[오류] 비교할 일자 데이터가 부족합니다 (최소 2일 필요).")
        sys.exit()

    # ── 3. 일일 경험치 계산 (레벨업 보정 포함) ─────────────────────────────
    print("[진행] 일일 경험치 획득량 계산 중 (레벨업 보정 포함)...")
    daily_dict, daily_cols, _ = compute_daily_exp(df, dates)
    df = pd.concat([df, pd.DataFrame(daily_dict, index=df.index)], axis=1)

    # ── 4. 레벨 구간 분류 ─────────────────────────────────────────────────────
    print("[진행] 레벨 구간 분류 중...")
    target_lv_col = f'Lv_{SHOWCASE_DATE}'
    if target_lv_col not in df.columns:
        target_lv_col = lv_cols_sorted[-1]
        print(f"   [정보] SHOWCASE_DATE 레벨 컬럼 없음 → 최신 컬럼 사용: {target_lv_col}")

    df['segment'] = df[target_lv_col].apply(get_segment)
    df = df.dropna(subset=['segment'])

    is_lv300 = (df[target_lv_col] == 300)
    lv300_count = is_lv300.sum()
    if lv300_count > 0:
        print(f"   [정보] 300레벨 유저 {lv300_count}명 감지 → 유령 유저 필터 제외 처리")

    # ── 5. 쇼케이스 전후 평균 계산 ────────────────────────────────────────────
    print("[진행] 쇼케이스 전후 평균 계산 중...")
    # Daily_YYYY-MM-DD 컬럼 = (DD-1)일 수집값 → DD일 수집값 증분.
    # 넥슨 API 는 새벽 갱신(전날 24시까지 누적)이므로
    # Daily_2025-12-13 이 담는 활동은 실제로 **12/12(쇼케이스 전날) 활동**.
    # 필터는 Daily_컬럼명(수집일) 기준이므로 쇼케이스 당일(12/13) 활동은
    # Daily_2025-12-14 에 담기고 Post 로 분류된다.
    showcase_dt = pd.to_datetime(SHOWCASE_DATE)
    pre_cols  = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
    post_cols = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) >  showcase_dt]

    # 0(비접속)은 포함, NaN(수집 공백/신규 진입 이전)은 제외
    df['Pre_Avg']  = df[pre_cols].mean(axis=1)  if pre_cols  else np.nan
    df['Post_Avg'] = df[post_cols].mean(axis=1) if post_cols else np.nan

    df['Pre_Valid_Days']  = df[pre_cols].notna().sum(axis=1)  if pre_cols  else 0
    df['Post_Valid_Days'] = df[post_cols].notna().sum(axis=1) if post_cols else 0

    # ── 6. 필터 1 : 전 기간 경험치 합 0 또는 전부 NaN 유저 제거 ──────────────
    # 300레벨은 경험치가 고정(0)이므로 제외
    daily_sum  = df[daily_cols].sum(axis=1, min_count=1)
    ghost_mask = (daily_sum == 0) & (~is_lv300)
    nan_mask   = daily_sum.isna()

    before = len(df)
    df = df[~ghost_mask & ~nan_mask]
    print(f"   [필터 1] 전 기간 경험치 0 유저 {before - len(df)}명 제거 → 잔여 {len(df)}명")

    # ── 7. 필터 2 : 14일+ 연속 경험치 0 유저 제거 (이탈 유저, 300레벨 제외) ──
    # NaN 은 비접속으로 보지 않으므로 fillna(False) 후 슬라이딩 윈도우 적용
    is_lv300_now = (df[target_lv_col] == 300)
    df_300  = df[is_lv300_now]
    df_rest = df[~is_lv300_now]

    zero_mat     = (df_rest[daily_cols] == 0).fillna(False).to_numpy()
    inactive_arr = np.zeros(len(df_rest), dtype=bool)
    for i in range(zero_mat.shape[1] - INACTIVE_THRESHOLD + 1):
        inactive_arr |= zero_mat[:, i:i + INACTIVE_THRESHOLD].all(axis=1)
    inactive_mask = pd.Series(inactive_arr, index=df_rest.index)

    before   = len(df)
    df_rest  = df_rest[~inactive_mask]
    df       = pd.concat([df_rest, df_300], ignore_index=True)
    print(f"   [필터 2] {INACTIVE_THRESHOLD}일+ 연속 비활성 유저 {before - len(df)}명 제거 → 잔여 {len(df)}명")

    # ── 8. 저장 ───────────────────────────────────────────────────────────────
    print("[저장] 분석용 데이터 저장 중...")
    cols_to_save = (
        ['name', 'job', 'world', 'segment',
         'Pre_Avg', 'Post_Avg',
         'Pre_Valid_Days', 'Post_Valid_Days']
        + daily_cols
    )
    df[cols_to_save].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n[완료] 전처리 성공! 총 {len(df):,}명 데이터 변환 완료.")
    print(f"[정보] 저장 위치: {OUTPUT_FILE}")
    print(f"[정보] Pre  기간: {pre_cols[0] if pre_cols else 'N/A'} ~ {pre_cols[-1] if pre_cols else 'N/A'} ({len(pre_cols)}일)")
    print(f"[정보] Post 기간: {post_cols[0] if post_cols else 'N/A'} ~ {post_cols[-1] if post_cols else 'N/A'} ({len(post_cols)}일)")


if __name__ == "__main__":
    preprocess_for_analysis()
