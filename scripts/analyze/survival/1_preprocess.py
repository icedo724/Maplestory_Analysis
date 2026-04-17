import pandas as pd
import numpy as np
import os
import sys

# ================= CONFIG =================
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR       = os.path.join(BASE_DIR, "data", "processed", "survival")

TRACKING_FILE   = os.path.join(BASE_DIR, "data", "raw", "daily_tracking_lv.csv")
USER_DETAIL_CSV = os.path.join(BASE_DIR, "data", "raw", "user_detail.csv")
LOG_FILE        = os.path.join(BASE_DIR, "data", "raw", "completed_log.txt")
OUTPUT_FILE     = os.path.join(OUT_DIR, "survival_data.csv")

SHOWCASE_DATE   = "2025-12-13"
CHURN_THRESHOLD = 7   # 연속 비활성 일수 기준

# 레벨별 요구 경험치 (레벨업 보정용, showcase/1_preprocess.py와 동일)
LEVEL_REQ_EXP = {
    284:  49_263_453_722_414,
    285:  99_512_176_519_276,
    286: 109_463_394_171_203,
    287: 120_409_733_588_323,
    288: 132_450_706_947_155,
    289: 145_695_777_641_870,
    290: 294_305_470_836_577,
    291: 323_736_017_920_234,
    292: 356_109_619_712_257,
    293: 391_720_581_683_482,
    294: 430_892_639_851_830,
    295: 870_403_132_500_696,
    296: 957_443_445_750_765,
    297: 1_053_187_790_325_841,
    298: 1_158_506_569_358_425,
    299: 1_737_759_854_037_637,
}
# ==========================================


def get_segment(lv):
    if pd.isna(lv):
        return None
    lv = int(lv)
    if 285 <= lv <= 289: return 'Lv.285~289'
    if 290 <= lv <= 294: return 'Lv.290~294'
    if 295 <= lv <= 299: return 'Lv.295~299'
    if lv == 300:        return 'Lv.300'
    return None


def load_completed_dates():
    if not os.path.exists(LOG_FILE):
        print("   [주의] completed_log.txt 없음 → 날짜 필터 미적용")
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


if __name__ == "__main__":
    # ── 1. 데이터 로드 ─────────────────────────────────────────────────────────
    for path in [TRACKING_FILE, USER_DETAIL_CSV]:
        if not os.path.exists(path):
            print(f"[오류] 파일 없음: {path}")
            sys.exit()

    print("[진행] 데이터 로드 중...")
    tracking    = pd.read_csv(TRACKING_FILE).drop_duplicates(subset='name')
    user_detail = pd.read_csv(USER_DETAIL_CSV)

    # ── 2. Inner join (user_detail 수집 대상만) ────────────────────────────────
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

    completed = load_completed_dates()
    if completed:
        before = len(dates)
        dates = [d for d in dates if d in completed]
        removed = before - len(dates)
        if removed:
            print(f"   [정보] 미완료 날짜 {removed}일 제거 → 사용 날짜: {len(dates)}일 "
                  f"(마지막: {dates[-1]})")
        exp_cols = [f'Exp_{d}' for d in dates]
        lv_cols  = [f'Lv_{d}'  for d in dates]

    if len(dates) < 2:
        print("[오류] 비교할 일자 부족 (최소 2일 필요).")
        sys.exit()

    # ── 4. Daily 경험치 계산 (레벨업 보정 포함) ───────────────────────────────
    print("[진행] 일일 경험치 계산 중 (레벨업 보정 포함)...")

    daily_dict   = {}
    daily_cols   = []
    levelup_total = 0

    for i in range(1, len(dates)):
        prev_d, curr_d = dates[i - 1], dates[i]
        col_name = f'Daily_{curr_d}'

        prev_exp = df[f'Exp_{prev_d}']
        curr_exp = df[f'Exp_{curr_d}']
        prev_lv  = df.get(f'Lv_{prev_d}')
        curr_lv  = df.get(f'Lv_{curr_d}')

        # API 미갱신 감지 (95%+ 동일 → 해당 날 제외)
        both = pd.concat([prev_exp.rename('p'), curr_exp.rename('c')], axis=1).dropna()
        if len(both) > 100 and (both['p'] == both['c']).mean() > 0.95:
            print(f"   [제외] {curr_d}: API 미갱신 → Daily 제외")
            continue

        diff = curr_exp - prev_exp

        if prev_lv is not None and curr_lv is not None:
            levelup_mask = (diff < 0) & (curr_lv > prev_lv) & prev_lv.notna() & curr_lv.notna()
            if levelup_mask.sum() > 0:
                levelup_total += int(levelup_mask.sum())
                req_exp   = prev_lv.map(LEVEL_REQ_EXP)
                corrected = req_exp - prev_exp + curr_exp
                diff      = diff.where(~levelup_mask, other=corrected)
        else:
            levelup_mask = pd.Series(False, index=diff.index)

        neg_mask = diff < 0
        daily_dict[col_name] = diff.where(~neg_mask.fillna(False), other=0)
        daily_cols.append(col_name)

    if levelup_total > 0:
        print(f"   [정보] 레벨업 보정: {levelup_total:,}건")

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

    # ── 6. 고스트 유저 제거 (전 기간 경험치 합 0, 300레벨 제외) ──────────────
    is_lv300   = (df[target_lv_col] == 300)
    daily_sum  = df[daily_cols].sum(axis=1, min_count=1)
    ghost_mask = (daily_sum == 0) & (~is_lv300)
    nan_mask   = daily_sum.isna()

    before = len(df)
    df = df[~ghost_mask & ~nan_mask].reset_index(drop=True)
    print(f"   [필터] 고스트 유저 {before - len(df)}명 제거 → 잔여 {len(df)}명")
    # ※ showcase와 달리 14일 연속 비활성 필터 미적용 (이탈 유저 포함)

    # ── 7. Pre/Post 평균 (쇼케이스 반응도 공변량) ────────────────────────────
    showcase_dt = pd.to_datetime(SHOWCASE_DATE)
    pre_cols  = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
    post_cols = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) >  showcase_dt]
    df['pre_avg']  = df[pre_cols].mean(axis=1)  if pre_cols  else np.nan
    df['post_avg'] = df[post_cols].mean(axis=1) if post_cols else np.nan

    # ── 8. 생존 분석 Feature 계산 (벡터화) ───────────────────────────────────
    print("[진행] 생존 분석 Feature 계산 중...")

    daily_mat       = df[daily_cols].to_numpy(dtype=float)
    active_mat_bool = daily_mat > 0
    valid_mat_bool  = ~np.isnan(daily_mat)

    has_any_active = active_mat_bool.any(axis=1)
    has_any_valid  = valid_mat_bool.any(axis=1)
    n_days         = len(daily_cols)
    dates_dt       = [pd.to_datetime(c.replace('Daily_', '')) for c in daily_cols]

    first_active_idx = np.where(has_any_active, np.argmax(active_mat_bool, axis=1), -1)
    last_active_idx  = np.where(has_any_active,
                                n_days - 1 - np.argmax(active_mat_bool[:, ::-1], axis=1), -1)
    last_valid_idx   = np.where(has_any_valid,
                                n_days - 1 - np.argmax(valid_mat_bool[:, ::-1], axis=1), -1)

    trailing_inactive = np.where(
        has_any_active & has_any_valid,
        last_valid_idx - last_active_idx,
        np.nan,
    ).astype(float)

    event_flag = np.where(
        has_any_active,
        (trailing_inactive >= CHURN_THRESHOLD).astype(float),
        np.nan,
    )

    # 이탈: first_active → last_active / 관측중단: first_active → last_valid
    duration_days = np.where(
        has_any_active & has_any_valid,
        np.where(
            event_flag == 1,
            last_active_idx - first_active_idx,
            last_valid_idx  - first_active_idx,
        ),
        np.nan,
    ).astype(float)
    duration_days = np.maximum(duration_days, 1)

    active_day_count = active_mat_bool.sum(axis=1).astype(float)
    total_valid_days = valid_mat_bool.sum(axis=1).astype(float)
    active_day_ratio = np.where(total_valid_days > 0,
                                active_day_count / total_valid_days, np.nan)

    active_exp_only = np.where(active_mat_bool, daily_mat, np.nan)
    with np.errstate(all='ignore'):
        avg_exp_on_active_days = np.nanmean(active_exp_only, axis=1)
    avg_exp_on_active_days = np.where(has_any_active, avg_exp_on_active_days, np.nan)

    # 캐릭터 나이 (date_create → last_valid_date)
    df['date_create'] = pd.to_datetime(df['date_create'], errors='coerce')
    last_valid_dates  = pd.to_datetime([
        dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else pd.NaT
        for i in last_valid_idx
    ])
    character_age_days = (last_valid_dates - df['date_create'].values).days

    first_active_date = [
        dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else None for i in first_active_idx
    ]
    last_active_date = [
        dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else None for i in last_active_idx
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
    result['avg_exp_on_active_days'] = avg_exp_on_active_days
    result['pre_avg']                = df['pre_avg'].values
    result['post_avg']               = df['post_avg'].values

    # 활성 기록 없는 유저 제거
    before = len(result)
    result = result[result['event_flag'].notna()].reset_index(drop=True)
    if before > len(result):
        print(f"   [정보] 활성 기록 없는 유저 {before - len(result)}명 추가 제거")

    # ── 10. 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    n_event    = int(result['event_flag'].sum())
    n_censored = len(result) - n_event
    print(f"\n[완료] {len(result):,}명 저장")
    print(f"       이탈(event=1): {n_event:,}명  |  관측중단(event=0): {n_censored:,}명")
    print(f"       이탈율: {n_event / len(result) * 100:.1f}%")
    print(f"       저장 위치: {OUTPUT_FILE}")
