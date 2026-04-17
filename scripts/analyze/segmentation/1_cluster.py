import numpy as np
import os
import sys

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# scripts/analyze/ 를 경로에 추가하여 utils 임포트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_segment, filter_completed_dates, compute_daily_exp

# ================= CONFIG =================
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR       = os.path.join(BASE_DIR, "data", "processed", "segmentation")

TRACKING_FILE   = os.path.join(BASE_DIR, "data", "raw", "daily_tracking_lv.csv")
USER_DETAIL_CSV = os.path.join(BASE_DIR, "data", "raw", "user_detail.csv")
LOG_FILE        = os.path.join(BASE_DIR, "data", "raw", "completed_log.txt")
OUTPUT_FILE     = os.path.join(OUT_DIR, "clustered_users.csv")

SHOWCASE_DATE = "2025-12-13"
K_RANGE       = range(2, 8)   # k=2..7 탐색 (원래 3~5 에서 확장)
RANDOM_SEED   = 42
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
    # access_flag, tier 는 피처로 직접 사용하지 않지만 출력 파일에 보존해 둔다.
    # 후속 분석(생존·스펙 분포)에서 서브그룹 필터로 활용 가능.
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
    df = df.dropna(subset=['segment']).reset_index(drop=True)
    if before > len(df):
        print(f"   [정보] 세그먼트 범위 외 {before - len(df)}명 제거")

    # ── 6. 고스트 유저 제거 ───────────────────────────────────────────────────
    # 전 기간 경험치 합이 0이거나 전부 NaN인 유저 제거 (300레벨 제외)
    # ※ 14일 연속 비활성 필터 미적용 — 이탈 패턴도 클러스터링 대상에 포함
    is_lv300  = (df[target_lv_col] == 300)
    daily_sum = df[daily_cols].sum(axis=1, min_count=1)
    ghost_mask = (daily_sum == 0) & (~is_lv300)
    nan_mask   = daily_sum.isna()

    before = len(df)
    df = df[~ghost_mask & ~nan_mask].reset_index(drop=True)
    print(f"   [필터] 고스트 유저 {before - len(df)}명 제거 → 잔여 {len(df)}명")

    # ── 7. 대칭 Pre/Post 평균 (쇼케이스 반응도 공변량) ──────────────────────
    # showcase/2_aggregate.py 와 동일한 대칭 기간 계산 방식 사용.
    # Daily_date 컬럼이 나타내는 실제 활동일 = Daily_date - 1일.
    showcase_dt = pd.to_datetime(SHOWCASE_DATE)
    activity_dates = [
        pd.to_datetime(c.replace('Daily_', '')) - pd.Timedelta(days=1)
        for c in daily_cols
    ]
    min_date = min(activity_dates)
    max_date = max(activity_dates)
    sym_days = min((showcase_dt - min_date).days, (max_date - showcase_dt).days)
    sym_start = showcase_dt - pd.Timedelta(days=sym_days)
    sym_end   = showcase_dt + pd.Timedelta(days=sym_days)

    # 대칭 기간 내 pre/post 컬럼 선택 (showcase/2_aggregate.py 기준과 동일)
    pre_cols  = [c for c in daily_cols
                 if sym_start < pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
    post_cols = [c for c in daily_cols
                 if showcase_dt < pd.to_datetime(c.replace('Daily_', '')) <= sym_end]

    df['pre_avg']  = df[pre_cols].mean(axis=1)  if pre_cols  else np.nan
    df['post_avg'] = df[post_cols].mean(axis=1) if post_cols else np.nan
    print(f"   [정보] Pre/Post 대칭 기간 ±{sym_days}일 "
          f"({sym_start.strftime('%Y-%m-%d')} ~ {sym_end.strftime('%Y-%m-%d')})")

    # ── 8. Feature 계산 ───────────────────────────────────────────────────────
    print("[진행] Feature 계산 중...")

    daily_mat       = df[daily_cols].to_numpy(dtype=float)
    active_mat_bool = daily_mat > 0
    valid_mat_bool  = ~np.isnan(daily_mat)

    active_day_count = active_mat_bool.sum(axis=1).astype(float)
    total_valid_days = valid_mat_bool.sum(axis=1).astype(float)
    active_day_ratio = np.where(
        total_valid_days > 0,
        active_day_count / total_valid_days,
        np.nan
    )

    # 활동일의 평균 경험치 (비활동일 NaN 처리 후 nanmean)
    active_exp_only = np.where(active_mat_bool, daily_mat, np.nan)
    with np.errstate(all='ignore'):
        avg_exp_on_active = np.nanmean(active_exp_only, axis=1)
    avg_exp_on_active = np.where(active_day_count > 0, avg_exp_on_active, np.nan)

    # 캐릭터 나이 (date_create → 마지막 유효 날짜)
    df['date_create'] = pd.to_datetime(df['date_create'], errors='coerce')
    daily_dates_dt = [pd.to_datetime(c.replace('Daily_', '')) for c in daily_cols]

    last_valid_idx = np.where(
        valid_mat_bool.any(axis=1),
        len(daily_cols) - 1 - np.argmax(valid_mat_bool[:, ::-1], axis=1),
        -1
    )
    last_valid_dates = pd.to_datetime([
        daily_dates_dt[int(i)].strftime('%Y-%m-%d') if i >= 0 else pd.NaT
        for i in last_valid_idx
    ])
    age_td = last_valid_dates - df['date_create'].values
    character_age_days = age_td.days.astype(float)

    # NaT(date_create/last_valid 누락) → .days 는 iNaT(거대 음수) 반환 → 명시 NaN 처리
    #   NaN 으로 두면 이후 feat_df.dropna() 단계에서 자연스럽게 클러스터링 대상에서 제외된다.
    nat_mask = df['date_create'].isna().to_numpy() | pd.isna(last_valid_dates).to_numpy()
    if nat_mask.any():
        print(f"   [경고] date_create/last_valid 누락 {int(nat_mask.sum())}건 → NaN 처리 (dropna 대상)")
        character_age_days = np.where(nat_mask, np.nan, character_age_days)

    # 음수 나이 방어 (date_create 데이터 오류로 인한 역전 방지, 매우 드묾)
    # ※ NaN 은 (< 0) 비교에서 False 로 빠지므로 위 nat_mask 처리와 충돌하지 않음
    neg_age_count = int((character_age_days < 0).sum())
    if neg_age_count > 0:
        print(f"   [경고] character_age_days 음수 {neg_age_count}건 → 0 으로 보정 (date_create 오류)")
        character_age_days = np.where(character_age_days < 0, 0, character_age_days)

    # avg_exp_pct : 세그먼트 내 퍼센타일 (0~100)
    # 세그먼트별 기저 활동량 차이를 제거하고 "상대적 열정도"를 측정
    df['_avg_exp']     = avg_exp_on_active
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

    # NaN 드롭 전 컬럼별 결측 현황 리포트
    before     = len(feat_df)
    nan_counts = feat_df.isna().sum()
    feat_df    = feat_df.dropna()
    dropped    = before - len(feat_df)
    if dropped > 0:
        print(f"   [정보] NaN 으로 제거된 유저: {dropped}명 (컬럼별 결측)")
        for col, cnt in nan_counts[nan_counts > 0].items():
            print(f"          ├ {col}: {cnt}명")
        print(f"          → 클러스터링 대상: {len(feat_df)}명")

    X        = feat_df.values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 10. 최적 모델·k 탐색 (KMeans vs GMM, 실루엣 기준) ───────────────────
    # 실루엣 점수: 높을수록 클러스터 간 경계가 명확.
    # KMeans  : 구형 클러스터 가정, 빠름.
    # GMM     : 타원형 클러스터 허용, 확률적 소속, BIC 로도 평가 가능.
    # 두 모델 중 실루엣이 높은 (모델, k) 조합을 최종 선택한다.
    print(f"[진행] 최적 k 탐색 중 (k={K_RANGE.start}~{K_RANGE.stop - 1}, KMeans vs GMM) ...")
    hdr = f"   {'k':>3} │ {'KMeans sil':>11} │ {'KMeans inertia':>16} │ {'GMM sil':>9} │ {'GMM BIC':>13}"
    print(hdr)
    print("   " + "─" * (len(hdr) - 3))

    results = []
    for k in K_RANGE:
        # KMeans
        km     = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km_lbl = km.fit_predict(X_scaled)
        km_sil = silhouette_score(X_scaled, km_lbl) if k > 1 else 0.0

        # GMM (n_init=5 로 수렴 안정성 확보)
        gmm     = GaussianMixture(n_components=k, random_state=RANDOM_SEED, n_init=5,
                                  covariance_type='full')
        gmm_lbl = gmm.fit_predict(X_scaled)
        gmm_sil = silhouette_score(X_scaled, gmm_lbl) if k > 1 else 0.0
        gmm_bic = gmm.bic(X_scaled)

        results.append({
            'k': k,
            'km': km,   'km_lbl': km_lbl,   'km_sil': km_sil,
            'gmm': gmm, 'gmm_lbl': gmm_lbl, 'gmm_sil': gmm_sil, 'gmm_bic': gmm_bic,
        })
        print(f"   {k:>3} │ {km_sil:>11.4f} │ {km.inertia_:>16,.0f} │ {gmm_sil:>9.4f} │ {gmm_bic:>13,.0f}")

    # 최고 실루엣 (모델 × k) 조합 선택
    best = max(results, key=lambda r: max(r['km_sil'], r['gmm_sil']))
    if best['km_sil'] >= best['gmm_sil']:
        best_k, best_model = best['k'], 'KMeans'
        labels, best_sil   = best['km_lbl'], best['km_sil']
    else:
        best_k, best_model = best['k'], 'GMM'
        labels, best_sil   = best['gmm_lbl'], best['gmm_sil']

    print(f"\n   → 최적 모델: {best_model},  k={best_k},  silhouette={best_sil:.4f}")
    if best_sil < 0.25:
        print("   [주의] silhouette < 0.25 — 클러스터 경계가 불명확합니다. "
              "피처 추가 또는 다른 알고리즘 검토를 권장합니다.")

    # ── 11. 결과 조립 ─────────────────────────────────────────────────────────
    feat_df = feat_df.copy()
    feat_df['cluster'] = labels

    result = df.loc[feat_df.index, [
        'name', 'segment', 'job', 'world', 'world_group',
        'tier', 'union_level', 'access_flag'
    ]].copy()
    result['active_day_ratio']   = feat_df['active_day_ratio'].values
    result['avg_exp_pct']        = feat_df['avg_exp_pct'].round(2).values
    result['avg_exp_on_active']  = df.loc[feat_df.index, '_avg_exp'].values
    result['character_age_days'] = feat_df['character_age_days'].values
    result['cluster']            = feat_df['cluster'].values
    result['pre_avg']            = df.loc[feat_df.index, 'pre_avg'].values
    result['post_avg']           = df.loc[feat_df.index, 'post_avg'].values

    # ── 12. 클러스터 프로파일 출력 ───────────────────────────────────────────
    pd.set_option('display.float_format', '{:,.2f}'.format)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)

    print("\n[클러스터 프로파일 — 피처 평균·중앙값]")
    profile_cols = ['active_day_ratio', 'avg_exp_pct', 'union_level',
                    'character_age_days', 'avg_exp_on_active']
    profile = (
        result.groupby('cluster')[profile_cols]
        .agg(['mean', 'median', 'count'])
    )
    print(profile)

    print("\n[클러스터별 Pre/Post 변화 — 쇼케이스 반응도]")
    pre_post = result.groupby('cluster')[['pre_avg', 'post_avg']].mean()
    pre_post['change_%'] = (
        (pre_post['post_avg'] - pre_post['pre_avg']) / pre_post['pre_avg'] * 100
    )
    print(pre_post.to_string(float_format='{:,.1f}'.format))

    print("\n[세그먼트 × 클러스터 분포]")
    print(result.groupby(['segment', 'cluster']).size().unstack(fill_value=0))

    # ── 13. 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n[완료] {len(result):,}명 저장  (모델={best_model}, k={best_k}, sil={best_sil:.4f})")
    print(f"       저장 위치: {OUTPUT_FILE}")
    print(f"       다음 단계: segmentation/2_profile.py 로 세부 프로파일 확인")
