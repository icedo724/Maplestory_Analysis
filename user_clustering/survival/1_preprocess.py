import numpy as np
import os
import sys

import pandas as pd

# ================= CONFIG =================
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# scripts/ 를 경로에 추가하여 utils 임포트
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from utils import get_segment, filter_completed_dates, compute_daily_exp

OUT_DIR       = os.path.join(BASE_DIR, "data", "processed", "survival")

TRACKING_FILE   = os.path.join(BASE_DIR, "data", "raw", "daily_tracking_lv.csv")
USER_DETAIL_CSV = os.path.join(BASE_DIR, "data", "raw", "user_detail.csv")
LOG_FILE        = os.path.join(BASE_DIR, "data", "raw", "completed_log.txt")
OUTPUT_FILE     = os.path.join(OUT_DIR, "survival_data.csv")

SHOWCASE_DATE = "2025-12-13"

# 이탈 기준: 7일 연속 비활성
# Lv.285+ 고레벨 유저는 일상적으로 매일 또는 격일 사냥하는 경향이 강하므로
# 7일 연속 비접속은 사실상 휴면·이탈 신호로 해석한다.
CHURN_THRESHOLD = 7
# ==========================================


if __name__ == "__main__":

    # ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
    for path in [TRACKING_FILE, USER_DETAIL_CSV]:
        if not os.path.exists(path):
            print(f"[오류] 파일 없음: {path}")
            sys.exit()

    print("[진행] 데이터 로드 중...")
    tracking    = pd.read_csv(TRACKING_FILE).drop_duplicates(subset='name')
    user_detail = pd.read_csv(USER_DETAIL_CSV)

    # ── 2. Inner join ──────────────────────────────────────────────────────────
    detail_cols = ['name', 'world_group', 'tier', 'union_level', 'date_create', 'access_flag']
    df = tracking.merge(user_detail[detail_cols], on='name', how='inner')
    print(f"[정보] tracking: {len(tracking):,}명 / user_detail: {len(user_detail):,}명 "
          f"→ inner join: {len(df):,}명")

    # ── 3. 날짜 컬럼 정렬 + 완료일 필터 ──────────────────────────────────────
    exp_cols = sorted(
        [c for c in df.columns if c.startswith('Exp_')],
        key=lambda x: pd.to_datetime(x.replace('Exp_', ''))
    )
    lv_cols = sorted(
        [c for c in df.columns if c.startswith('Lv_')],
        key=lambda x: pd.to_datetime(x.replace('Lv_', ''))
    )
    dates = [c.replace('Exp_', '') for c in exp_cols]
    dates = filter_completed_dates(dates, LOG_FILE)
    lv_cols = [f'Lv_{d}' for d in dates]   # 동기화

    if len(dates) < 2:
        print("[오류] 비교할 일자 부족 (최소 2일 필요).")
        sys.exit()

    # ── 4. Daily 경험치 계산 ──────────────────────────────────────────────────
    print("[진행] 일일 경험치 계산 중 (레벨업 보정 포함)...")
    daily_dict, daily_cols, _ = compute_daily_exp(df, dates)
    if not daily_cols:
        print("[오류] 유효한 Daily 컬럼이 없습니다. "
              "모든 날짜가 API 미갱신으로 제외됐거나 dates 길이가 1 이하입니다.")
        sys.exit()
    df = pd.concat([df, pd.DataFrame(daily_dict, index=df.index)], axis=1)

    # ── 5. Segment 분류 ───────────────────────────────────────────────────────
    target_lv_col = f'Lv_{SHOWCASE_DATE}'
    if target_lv_col not in df.columns:
        target_lv_col = lv_cols[-1]
        print(f"   [정보] SHOWCASE_DATE 레벨 컬럼 없음 → 최신 컬럼 사용: {target_lv_col}")

    df['segment'] = df[target_lv_col].apply(get_segment)
    before = len(df)
    df = df.dropna(subset=['segment'])
    if before > len(df):
        print(f"   [정보] 세그먼트 범위 외 {before - len(df)}명 제거")

    # ── 6. 고스트 유저 제거 ───────────────────────────────────────────────────
    # ※ showcase 와 달리 14일 연속 비활성 필터 미적용 — 이탈 유저 포함이 목적
    is_lv300   = (df[target_lv_col] == 300)
    daily_sum  = df[daily_cols].sum(axis=1, min_count=1)
    ghost_mask = (daily_sum == 0) & (~is_lv300)
    nan_mask   = daily_sum.isna()

    before = len(df)
    df = df[~ghost_mask & ~nan_mask].reset_index(drop=True)
    print(f"   [필터] 고스트 유저 {before - len(df)}명 제거 → 잔여 {len(df)}명")

    # ── 7. 대칭 Pre/Post 평균 (쇼케이스 반응도 공변량) ──────────────────────
    # showcase/2_aggregate.py 와 동일한 대칭 기간 계산 방식 사용
    showcase_dt = pd.to_datetime(SHOWCASE_DATE)
    activity_dates = [
        pd.to_datetime(c.replace('Daily_', '')) - pd.Timedelta(days=1)
        for c in daily_cols
    ]
    sym_days  = min((showcase_dt - min(activity_dates)).days,
                    (max(activity_dates) - showcase_dt).days)
    sym_start = showcase_dt - pd.Timedelta(days=sym_days)
    sym_end   = showcase_dt + pd.Timedelta(days=sym_days)

    pre_cols  = [c for c in daily_cols
                 if sym_start < pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
    post_cols = [c for c in daily_cols
                 if showcase_dt < pd.to_datetime(c.replace('Daily_', '')) <= sym_end]
    df['pre_avg']  = df[pre_cols].mean(axis=1)  if pre_cols  else np.nan
    df['post_avg'] = df[post_cols].mean(axis=1) if post_cols else np.nan

    # ── 8. 생존 분석 Feature 계산 ─────────────────────────────────────────────
    print("[진행] 생존 분석 Feature 계산 중...")

    daily_mat       = df[daily_cols].to_numpy(dtype=float)
    active_mat_bool = daily_mat > 0
    valid_mat_bool  = ~np.isnan(daily_mat)

    has_any_active = active_mat_bool.any(axis=1)
    has_any_valid  = valid_mat_bool.any(axis=1)
    n_days         = len(daily_cols)
    daily_dates_dt = [pd.to_datetime(c.replace('Daily_', '')) for c in daily_cols]

    first_active_idx = np.where(has_any_active, np.argmax(active_mat_bool, axis=1), -1)
    last_active_idx  = np.where(
        has_any_active,
        n_days - 1 - np.argmax(active_mat_bool[:, ::-1], axis=1),
        -1
    )
    last_valid_idx = np.where(
        has_any_valid,
        n_days - 1 - np.argmax(valid_mat_bool[:, ::-1], axis=1),
        -1
    )

    # 마지막 활성일 이후 ~ 마지막 유효 관측일까지 비활성 일수
    trailing_inactive = np.where(
        has_any_active & has_any_valid,
        last_valid_idx - last_active_idx,
        np.nan,
    ).astype(float)

    # 이탈 플래그: trailing_inactive >= CHURN_THRESHOLD
    event_flag = np.where(
        has_any_active,
        (trailing_inactive >= CHURN_THRESHOLD).astype(float),
        np.nan,
    )

    # Lv.300 도달 유저는 경험치가 오르지 않는 구조적 특성 → 이탈로 판정하지 않고 관측중단 처리
    last_lv_col = lv_cols[-1]
    lv_at_end   = df[last_lv_col].values
    lv300_mask  = (lv_at_end == 300)
    if lv300_mask.sum() > 0:
        print(f"   [정보] 관측 기간 중 Lv.300 도달 {int(lv300_mask.sum())}건 → 관측중단(event=0) 처리")
        event_flag = np.where(lv300_mask, 0, event_flag)

    # duration: 이탈 → first_active ~ last_active / 관측중단 → first_active ~ last_valid
    duration_days = np.where(
        has_any_active & has_any_valid,
        np.where(
            event_flag == 1,
            last_active_idx - first_active_idx,
            last_valid_idx  - first_active_idx,
        ),
        np.nan,
    ).astype(float)
    # 최소 1일 보장 (당일 이탈 방지)
    duration_days = np.maximum(duration_days, 1)

    active_day_count = active_mat_bool.sum(axis=1).astype(float)
    total_valid_days = valid_mat_bool.sum(axis=1).astype(float)
    active_day_ratio = np.where(
        total_valid_days > 0,
        active_day_count / total_valid_days,
        np.nan
    )

    active_exp_only = np.where(active_mat_bool, daily_mat, np.nan)
    with np.errstate(all='ignore'):
        avg_exp_on_active = np.nanmean(active_exp_only, axis=1)
    avg_exp_on_active = np.where(has_any_active, avg_exp_on_active, np.nan)

    # 캐릭터 나이 (date_create → last_valid_date)
    df['date_create'] = pd.to_datetime(df['date_create'], errors='coerce')
    last_valid_dates  = pd.to_datetime([
        daily_dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else pd.NaT
        for i in last_valid_idx
    ])
    age_td = last_valid_dates - df['date_create'].values
    character_age_days = age_td.days.astype(float)
    # date_create / last_valid 누락(NaT) → .days 는 iNaT(거대 음수) 반환 → 명시 NaN 처리
    nat_mask = df['date_create'].isna().to_numpy() | np.array(pd.isna(last_valid_dates))
    if nat_mask.any():
        print(f"   [경고] date_create/last_valid 누락 {int(nat_mask.sum())}건 → NaN 처리")
        character_age_days = np.where(nat_mask, np.nan, character_age_days)

    first_active_date = [
        daily_dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else None
        for i in first_active_idx
    ]
    last_active_date = [
        daily_dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else None
        for i in last_active_idx
    ]

    # ── 9. 결과 조립 ──────────────────────────────────────────────────────────
    result = df[['name', 'segment', 'job', 'world', 'world_group', 'tier',
                 'union_level', 'access_flag']].copy()

    result['character_age_days']     = character_age_days
    result['event_flag']             = pd.array(
        [int(v) if not np.isnan(v) else pd.NA for v in event_flag], dtype='Int64'
    )
    result['duration_days']          = pd.Series(np.round(duration_days)).astype('Int64')
    result['trailing_inactive_days'] = pd.Series(np.round(trailing_inactive)).astype('Int64')
    result['first_active_date']      = first_active_date
    result['last_active_date']       = last_active_date
    result['active_day_count']       = active_day_count.astype(int)
    result['active_day_ratio']       = np.round(active_day_ratio, 4)
    result['avg_exp_on_active']      = avg_exp_on_active
    result['pre_avg']                = df['pre_avg'].values
    result['post_avg']               = df['post_avg'].values

    # 활성 기록이 아예 없는 유저, 또는 duration 계산 불가 유저 제거
    before = len(result)
    result = result[result['event_flag'].notna() & result['duration_days'].notna()].reset_index(drop=True)
    if before > len(result):
        print(f"   [정보] 활성 기록 없는 유저 {before - len(result)}명 추가 제거")

    # ── 10. 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    n_event    = int(result['event_flag'].sum())
    n_censored = len(result) - n_event
    print(f"\n[완료] {len(result):,}명 저장  (이탈 기준: {CHURN_THRESHOLD}일 연속 비활성)")
    print(f"       이탈(event=1)   : {n_event:,}명")
    print(f"       관측중단(event=0): {n_censored:,}명")
    print(f"       이탈율          : {n_event / len(result) * 100:.1f}%")
    print(f"       저장 위치       : {OUTPUT_FILE}")
    print(f"       다음 단계       : survival/2_analyze.py 로 KM 곡선·로그랭크 검정 실행")
