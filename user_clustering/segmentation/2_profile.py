"""
클러스터 프로파일 분석
— segmentation/1_cluster.py 출력(clustered_users.csv)을 기반으로 군집 특성을 상세 출력한다.

실행 순서: 1_cluster.py → 2_profile.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')

# ================= CONFIG =================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "segmentation", "clustered_users.csv")
# ==========================================

SEP  = "=" * 72
SEP2 = "-" * 72

FEAT_COLS = ['active_day_ratio', 'avg_exp_pct', 'union_level',
             'character_age_days', 'avg_exp_on_active', 'stat_atk_pct']


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subsection(title):
    print(f"\n{SEP2}\n  {title}\n{SEP2}")


if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"[오류] 파일 없음: {INPUT_FILE}")
        print("       segmentation/1_cluster.py 를 먼저 실행하세요.")
        sys.exit()

    df = pd.read_csv(INPUT_FILE)
    n_clusters = df['cluster'].nunique()
    pd.set_option('display.float_format', '{:,.2f}'.format)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)

    print(f"[로드] {len(df):,}명  클러스터 수: {n_clusters}")

    # ══════════════════════════════════════════════════════════════════
    # [1] 클러스터 크기
    # ══════════════════════════════════════════════════════════════════
    section("[1] 클러스터 크기 분포")
    sizes = df['cluster'].value_counts().sort_index()
    for c, n in sizes.items():
        bar = "█" * int(n / len(df) * 40)
        print(f"  Cluster {c}: {n:>5}명 ({n / len(df) * 100:4.1f}%)  {bar}")

    # ══════════════════════════════════════════════════════════════════
    # [2] 피처 분포 (mean ± std / median)
    # ══════════════════════════════════════════════════════════════════
    section("[2] 클러스터별 피처 통계")
    for col in FEAT_COLS:
        subsection(col)
        profile = df.groupby('cluster')[col].agg(
            n='count', mean='mean', std='std',
            median='median', min='min', max='max'
        )
        print(profile.to_string())

    # ══════════════════════════════════════════════════════════════════
    # [3] 쇼케이스 반응 (Pre/Post)
    # ══════════════════════════════════════════════════════════════════
    section("[3] 클러스터별 쇼케이스 반응 — Pre / Post 평균")
    pre_post = df.groupby('cluster')[['pre_avg', 'post_avg']].mean()
    pre_post['change_%'] = (
        (pre_post['post_avg'] - pre_post['pre_avg']) / pre_post['pre_avg'] * 100
    )
    print(pre_post.to_string(float_format='{:,.1f}'.format))

    # ══════════════════════════════════════════════════════════════════
    # [4] 세그먼트 × 클러스터 교차 분포
    # ══════════════════════════════════════════════════════════════════
    section("[4] 세그먼트 × 클러스터 교차 분포")
    ct = pd.crosstab(df['segment'], df['cluster'], margins=True)
    print(ct.to_string())

    print("\n[세그먼트 내 클러스터 비율 (%)]")
    ct_pct = pd.crosstab(df['segment'], df['cluster'], normalize='index') * 100
    print(ct_pct.round(1).to_string())

    # ══════════════════════════════════════════════════════════════════
    # [5] 클러스터별 상위 직업 분포
    # ══════════════════════════════════════════════════════════════════
    section("[5] 클러스터별 상위 직업 분포 (Top 5)")
    for c in sorted(df['cluster'].unique()):
        sd = df[df['cluster'] == c]
        top_jobs = sd['job'].value_counts().head(5)
        print(f"\n  Cluster {c}  (n={len(sd):,}명)")
        for job, cnt in top_jobs.items():
            print(f"    {job:<22} {cnt:>4}명  ({cnt / len(sd) * 100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # [6] 월드 그룹 분포
    # ══════════════════════════════════════════════════════════════════
    section("[6] 클러스터 × 월드 그룹 분포 (%)")
    wg_ct = pd.crosstab(df['cluster'], df['world_group'], normalize='index') * 100
    print(wg_ct.round(1).to_string())

    # ══════════════════════════════════════════════════════════════════
    # [7] ANOVA: 클러스터 간 피처 차이 유의성
    # ══════════════════════════════════════════════════════════════════
    section("[7] 클러스터 간 피처 차이 — One-way ANOVA")
    print(f"  {'피처':<25}  {'F-stat':>9}  {'p-value':>12}  판정")
    print(f"  {'─'*25}  {'─'*9}  {'─'*12}  {'─'*14}")
    for col in FEAT_COLS:
        groups = [
            df[df['cluster'] == c][col].dropna().values
            for c in sorted(df['cluster'].unique())
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        sig = "★★★ p<0.001" if p_val < 0.001 else ("★   p<0.05 " if p_val < 0.05 else "비유의     ")
        print(f"  {col:<25}  {f_stat:>9.3f}  {p_val:>12.4e}  {sig}")

    # ══════════════════════════════════════════════════════════════════
    # [8] 클러스터 해석 힌트 (자동 레이블링)
    # ══════════════════════════════════════════════════════════════════
    section("[8] 클러스터 자동 레이블 힌트")
    cluster_means = df.groupby('cluster')[['active_day_ratio', 'avg_exp_pct', 'union_level']].mean()
    overall = cluster_means.mean()

    print(f"  전체 평균 — 활동일비율: {overall['active_day_ratio']:.2f} | "
          f"경험치퍼센타일: {overall['avg_exp_pct']:.1f} | 유니온: {overall['union_level']:.0f}\n")

    for c in cluster_means.index:
        row = cluster_means.loc[c]
        tags = []
        tags.append("고활동" if row['active_day_ratio'] > overall['active_day_ratio'] else "저활동")
        tags.append("강파밍" if row['avg_exp_pct']     > overall['avg_exp_pct']       else "저파밍")
        tags.append("고스펙" if row['union_level']     > overall['union_level']        else "저스펙")
        print(f"  Cluster {c}: {' / '.join(tags)}")
        print(f"    활동일비율={row['active_day_ratio']:.2f}  "
              f"경험치퍼센타일={row['avg_exp_pct']:.1f}  "
              f"유니온={row['union_level']:.0f}")

    print(f"\n{SEP}\n  분석 완료\n{SEP}")
