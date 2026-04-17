import pandas as pd
import numpy as np
import os
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ================= CONFIG =================
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR       = os.path.join(BASE_DIR, "data", "processed", "segmentation")

TRACKING_FILE   = os.path.join(BASE_DIR, "data", "raw", "daily_tracking_lv.csv")
USER_DETAIL_CSV = os.path.join(BASE_DIR, "data", "raw", "user_detail.csv")
LOG_FILE        = os.path.join(BASE_DIR, "data", "raw", "completed_log.txt")
OUTPUT_FILE     = os.path.join(OUT_DIR, "clustered_users.csv")

SHOWCASE_DATE = "2025-12-13"
K_RANGE       = range(3, 6)    # k=3,4,5 탐색
RANDOM_SEED   = 42

# 레벨별 요구 경험치 (레벨업 보정용)
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
    df = df.dropna(subset=['segment']).reset_index(drop=True)
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
    # ※ 14일 연속 비활성 필터 미적용 (이탈 패턴도 클러스터 대상)

    # ── 7. Pre/Post 평균 (쇼케이스 반응도 공변량) ────────────────────────────
    showcase_dt = pd.to_datetime(SHOWCASE_DATE)
    pre_cols  = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
    post_cols = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) >  showcase_dt]
    df['pre_avg']  = df[pre_cols].mean(axis=1)  if pre_cols  else np.nan
    df['post_avg'] = df[post_cols].mean(axis=1) if post_cols else np.nan

    # ── 8. Feature 계산 ───────────────────────────────────────────────────────
    print("[진행] Feature 계산 중...")

    daily_mat       = df[daily_cols].to_numpy(dtype=float)
    active_mat_bool = daily_mat > 0
    valid_mat_bool  = ~np.isnan(daily_mat)

    active_day_count = active_mat_bool.sum(axis=1).astype(float)
    total_valid_days = valid_mat_bool.sum(axis=1).astype(float)
    active_day_ratio = np.where(total_valid_days > 0,
                                active_day_count / total_valid_days, np.nan)

    active_exp_only = np.where(active_mat_bool, daily_mat, np.nan)
    with np.errstate(all='ignore'):
        avg_exp_on_active_days = np.nanmean(active_exp_only, axis=1)
    avg_exp_on_active_days = np.where(active_day_count > 0, avg_exp_on_active_days, np.nan)

    # 캐릭터 나이 (date_create → 마지막 유효일)
    df['date_create'] = pd.to_datetime(df['date_create'], errors='coerce')
    dates_dt = [pd.to_datetime(c.replace('Daily_', '')) for c in daily_cols]
    last_valid_idx = np.where(
        valid_mat_bool.any(axis=1),
        len(daily_cols) - 1 - np.argmax(valid_mat_bool[:, ::-1], axis=1),
        -1
    )
    last_valid_dates = pd.to_datetime([
        dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else pd.NaT
        for i in last_valid_idx
    ])
    character_age_days = (last_valid_dates - df['date_create'].values).days.astype(float)

    # avg_exp_pct: 세그먼트 내 퍼센타일 (0~100)
    df['_avg_exp']     = avg_exp_on_active_days
    df['_avg_exp_pct'] = np.nan
    for seg, grp in df.groupby('segment'):
        ranks = grp['_avg_exp'].rank(pct=True, na_option='keep') * 100
        df.loc[grp.index, '_avg_exp_pct'] = ranks

    df['_active_day_ratio']   = active_day_ratio
    df['_character_age_days'] = character_age_days

    # ── 9. Feature 행렬 구성 ──────────────────────────────────────────────────
    FEATURE_COLS = ['_active_day_ratio', '_avg_exp_pct', 'union_level', '_character_age_days']

    feat_df = df[FEATURE_COLS].copy()
    feat_df.columns = ['active_day_ratio', 'avg_exp_pct', 'union_level', 'character_age_days']

    before = len(feat_df)
    feat_df = feat_df.dropna()
    if before > len(feat_df):
        print(f"   [정보] NaN 유저 {before - len(feat_df)}명 제거 → {len(feat_df)}명 클러스터링")

    X        = feat_df.values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 10. 최적 k 탐색 (엘보우 + 실루엣) ───────────────────────────────────
    print("[진행] 최적 k 탐색 중...")
    inertia_list    = []
    silhouette_list = []

    for k in K_RANGE:
        km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia_list.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouette_list.append(sil)
        print(f"   k={k}: inertia={km.inertia_:,.0f} | silhouette={sil:.4f}")

    best_k = list(K_RANGE)[int(np.argmax(silhouette_list))]
    print(f"\n   → 최적 k (실루엣 기준): {best_k}")

    # ── 11. 최적 k로 최종 클러스터링 ─────────────────────────────────────────
    km_final         = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    feat_df          = feat_df.copy()
    feat_df['cluster'] = km_final.fit_predict(X_scaled)

    # ── 12. 결과 조립 ─────────────────────────────────────────────────────────
    result = df.loc[feat_df.index, ['name', 'segment', 'job', 'world', 'world_group',
                                    'tier', 'union_level', 'access_flag']].copy()
    result['active_day_ratio']   = feat_df['active_day_ratio'].values
    result['avg_exp_pct']        = feat_df['avg_exp_pct'].round(2).values
    result['avg_exp_on_active']  = df.loc[feat_df.index, '_avg_exp'].values
    result['character_age_days'] = feat_df['character_age_days'].values
    result['cluster']            = feat_df['cluster'].values
    result['pre_avg']            = df.loc[feat_df.index, 'pre_avg'].values
    result['post_avg']           = df.loc[feat_df.index, 'post_avg'].values

    # ── 13. 클러스터 프로파일 출력 ───────────────────────────────────────────
    print("\n[클러스터 프로파일]")
    profile_cols = ['active_day_ratio', 'avg_exp_pct', 'union_level', 'character_age_days',
                    'avg_exp_on_active']
    profile = (
        result.groupby('cluster')[profile_cols]
        .agg(['mean', 'median', 'count'])
    )
    pd.set_option('display.float_format', '{:,.2f}'.format)
    pd.set_option('display.max_columns', 20)
    print(profile)

    print("\n[세그먼트 × 클러스터 분포]")
    print(result.groupby(['segment', 'cluster']).size().unstack(fill_value=0))

    # ── 14. 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n[완료] {len(result):,}명 저장 (k={best_k})")
    print(f"       저장 위치: {OUTPUT_FILE}")
