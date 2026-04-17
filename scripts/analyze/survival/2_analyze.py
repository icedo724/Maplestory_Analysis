"""
Kaplan-Meier 생존 분석
— survival/1_preprocess.py 출력(survival_data.csv)을 기반으로
  KM 곡선 요약·로그랭크 검정·활동도 그룹 분석을 수행한다.

사전 조건
---------
  pip install lifelines

실행 순서: survival/1_preprocess.py → survival/2_analyze.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("[주의] lifelines 미설치 → KM 곡선 기능 비활성화")
    print("       pip install lifelines 후 재실행하세요.\n")

# ================= CONFIG =================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
INPUT_FILE   = os.path.join(BASE_DIR, "data", "processed", "survival", "survival_data.csv")
CLUSTER_FILE = os.path.join(BASE_DIR, "data", "processed", "segmentation", "clustered_users.csv")
# ==========================================

SEP  = "=" * 72
SEP2 = "-" * 72

SEGMENT_ORDER = ['Lv.285~289', 'Lv.290~294', 'Lv.295~299', 'Lv.300']


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subsection(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


def km_summary_line(durations, events, label="전체"):
    """
    KM 추정으로 중앙 생존 일수와 이탈율을 한 줄로 출력한다.
    lifelines 없으면 단순 통계만 출력.
    """
    n       = len(durations)
    ev_rate = events.mean() * 100

    if HAS_LIFELINES:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events, label=label)
        median = kmf.median_survival_time_
        # 30·60·90일 시점 생존율
        surv_30  = float(kmf.survival_function_at_times([30]).iloc[0])  if 30  <= durations.max() else np.nan
        surv_60  = float(kmf.survival_function_at_times([60]).iloc[0])  if 60  <= durations.max() else np.nan
        surv_90  = float(kmf.survival_function_at_times([90]).iloc[0])  if 90  <= durations.max() else np.nan
        print(f"  {label:<25} n={n:>5}  중앙생존={median:>6.1f}일  "
              f"이탈율={ev_rate:5.1f}%  "
              f"S(30)={surv_30:.3f}  S(60)={surv_60:.3f}  S(90)={surv_90:.3f}")
        return kmf
    else:
        median_dur = durations[events == 1].median() if events.sum() > 0 else np.nan
        print(f"  {label:<25} n={n:>5}  이탈자중앙duration={median_dur:.1f}일  이탈율={ev_rate:.1f}%")
        return None


def pairwise_logrank(df_sub, group_col, dur_col='duration_days', ev_col='event_flag'):
    """그룹 쌍별 로그랭크 검정 결과를 테이블 형식으로 출력."""
    groups = sorted(df_sub[group_col].dropna().unique())
    if len(groups) < 2:
        print("  [주의] 그룹이 2개 미만 — 검정 불가")
        return

    print(f"  {'비교':>36}  {'p-value':>12}  판정")
    print(f"  {'─'*36}  {'─'*12}  ──────")
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            d1 = df_sub[df_sub[group_col] == g1]
            d2 = df_sub[df_sub[group_col] == g2]
            try:
                r   = logrank_test(d1[dur_col], d2[dur_col], d1[ev_col], d2[ev_col])
                sig = "★★★" if r.p_value < 0.001 else ("★  " if r.p_value < 0.05 else "   ")
                print(f"  {str(g1):<18} vs {str(g2):<14}  {r.p_value:>12.4e}  {sig}")
            except Exception as e:
                print(f"  {str(g1)} vs {str(g2)}  [오류: {e}]")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[오류] 파일 없음: {INPUT_FILE}")
        print("       survival/1_preprocess.py 를 먼저 실행하세요.")
        sys.exit()

    df = pd.read_csv(INPUT_FILE)
    df = df[df['event_flag'].notna() & df['duration_days'].notna()].copy()
    if len(df) == 0:
        print("[오류] 유효한 event_flag / duration_days 데이터가 없습니다. "
              "survival/1_preprocess.py 출력을 확인하세요.")
        sys.exit()
    df['event_flag']    = df['event_flag'].astype(int)
    df['duration_days'] = df['duration_days'].astype(float)

    n_event    = df['event_flag'].sum()
    n_censored = len(df) - n_event
    print(f"[로드] 총 {len(df):,}명  "
          f"이탈(event=1): {n_event:,}명  "
          f"관측중단(event=0): {n_censored:,}명  "
          f"이탈율: {n_event / len(df) * 100:.1f}%")
    print(f"       duration 범위: {df['duration_days'].min():.0f}일 ~ {df['duration_days'].max():.0f}일")

    # 클러스터 정보 병합 (있으면)
    has_cluster = os.path.exists(CLUSTER_FILE)
    if has_cluster:
        clusters = pd.read_csv(CLUSTER_FILE)[['name', 'cluster']]
        df = df.merge(clusters, on='name', how='left')
        matched = df['cluster'].notna().sum()
        print(f"   [정보] 클러스터 병합: {matched:,}명 매칭")
    else:
        print("   [정보] clustered_users.csv 없음 → 클러스터별 분석 생략")

    dur  = df['duration_days']
    evt  = df['event_flag']

    # ══════════════════════════════════════════════════════════════════
    # [1] 전체 KM
    # ══════════════════════════════════════════════════════════════════
    section("[1] 전체 유저 생존 분석")
    print(f"  {'그룹':<25} {'n':>5}  {'중앙생존':>8}  {'이탈율':>6}  "
          f"{'S(30)':>7}  {'S(60)':>7}  {'S(90)':>7}")
    print(f"  {'─'*25} {'─'*5}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
    km_summary_line(dur, evt, "전체")

    # ══════════════════════════════════════════════════════════════════
    # [2] 레벨 구간별 KM + 로그랭크 검정
    # ══════════════════════════════════════════════════════════════════
    section("[2] 레벨 구간별 생존 분석")
    segs = [s for s in SEGMENT_ORDER if s in df['segment'].values]

    print(f"  {'그룹':<25} {'n':>5}  {'중앙생존':>8}  {'이탈율':>6}  "
          f"{'S(30)':>7}  {'S(60)':>7}  {'S(90)':>7}")
    print(f"  {'─'*25} {'─'*5}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
    for seg in segs:
        sd = df[df['segment'] == seg]
        km_summary_line(sd['duration_days'], sd['event_flag'], seg)

    if HAS_LIFELINES and len(segs) >= 2:
        subsection("2-1. 다중 로그랭크 검정 (레벨 구간 전체)")
        df_seg = df[df['segment'].isin(segs)].copy()
        try:
            r = multivariate_logrank_test(
                df_seg['duration_days'], df_seg['segment'], df_seg['event_flag']
            )
            sig = "유의미 (p<0.05)" if r.p_value < 0.05 else "비유의"
            print(f"  test_stat={r.test_statistic:.4f}  p-value={r.p_value:.4e}  → {sig}")
        except Exception as e:
            print(f"  [오류] {e}")

        subsection("2-2. 쌍별 로그랭크 검정 (레벨 구간)")
        pairwise_logrank(df_seg, 'segment')

    # ══════════════════════════════════════════════════════════════════
    # [3] 클러스터별 KM + 로그랭크 검정
    # ══════════════════════════════════════════════════════════════════
    if has_cluster and 'cluster' in df.columns:
        section("[3] 클러스터별 생존 분석")
        df_cl = df[df['cluster'].notna()].copy()
        df_cl['cluster'] = df_cl['cluster'].astype(int)
        cluster_ids = sorted(df_cl['cluster'].unique())

        print(f"  {'그룹':<25} {'n':>5}  {'중앙생존':>8}  {'이탈율':>6}  "
              f"{'S(30)':>7}  {'S(60)':>7}  {'S(90)':>7}")
        print(f"  {'─'*25} {'─'*5}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
        for c in cluster_ids:
            sd = df_cl[df_cl['cluster'] == c]
            km_summary_line(sd['duration_days'], sd['event_flag'], f"Cluster {c}")

        if HAS_LIFELINES and len(cluster_ids) >= 2:
            subsection("3-1. 다중 로그랭크 검정 (클러스터 전체)")
            try:
                r = multivariate_logrank_test(
                    df_cl['duration_days'],
                    df_cl['cluster'].astype(str),
                    df_cl['event_flag']
                )
                sig = "유의미 (p<0.05)" if r.p_value < 0.05 else "비유의"
                print(f"  test_stat={r.test_statistic:.4f}  p-value={r.p_value:.4e}  → {sig}")
            except Exception as e:
                print(f"  [오류] {e}")

            subsection("3-2. 쌍별 로그랭크 검정 (클러스터)")
            pairwise_logrank(df_cl, 'cluster')

    # ══════════════════════════════════════════════════════════════════
    # [4] 활동도 그룹별 생존 패턴
    # ══════════════════════════════════════════════════════════════════
    section("[4] 활동 일수 비율 그룹별 생존 분석")
    df['activity_group'] = pd.cut(
        df['active_day_ratio'],
        bins=[0.0, 0.25, 0.5, 0.75, 1.01],
        labels=['저활동 (0~25%)', '중저활동 (25~50%)', '중고활동 (50~75%)', '고활동 (75~100%)'],
        include_lowest=True
    )
    act_labels = ['저활동 (0~25%)', '중저활동 (25~50%)', '중고활동 (50~75%)', '고활동 (75~100%)']

    print(f"  {'그룹':<25} {'n':>5}  {'중앙생존':>8}  {'이탈율':>6}  "
          f"{'S(30)':>7}  {'S(60)':>7}  {'S(90)':>7}")
    print(f"  {'─'*25} {'─'*5}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
    for grp in act_labels:
        sd = df[df['activity_group'] == grp]
        if len(sd) < 5:
            continue
        km_summary_line(sd['duration_days'], sd['event_flag'], grp)

    if HAS_LIFELINES:
        df_act = df[df['activity_group'].notna()].copy()
        if df_act['activity_group'].nunique() >= 2:
            subsection("4-1. 다중 로그랭크 검정 (활동도 그룹)")
            try:
                r = multivariate_logrank_test(
                    df_act['duration_days'],
                    df_act['activity_group'].astype(str),
                    df_act['event_flag']
                )
                sig = "유의미 (p<0.05)" if r.p_value < 0.05 else "비유의"
                print(f"  test_stat={r.test_statistic:.4f}  p-value={r.p_value:.4e}  → {sig}")
            except Exception as e:
                print(f"  [오류] {e}")

    # ══════════════════════════════════════════════════════════════════
    # [5] 기술 통계 요약
    # ══════════════════════════════════════════════════════════════════
    section("[5] duration_days 기술 통계 (이탈 vs 관측중단)")
    for flag, label in [(1, "이탈 (event=1)"), (0, "관측중단 (event=0)")]:
        sd = df[df['event_flag'] == flag]['duration_days']
        if len(sd) == 0:
            continue
        print(f"\n  {label}  (n={len(sd):,})")
        print(f"    평균={sd.mean():.1f}일  중앙값={sd.median():.1f}일  "
              f"std={sd.std():.1f}  min={sd.min():.0f}  max={sd.max():.0f}")

    print(f"\n{SEP}\n  분석 완료\n{SEP}")
