import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import CLUSTER_FILE, SURVIVAL_FILE, STAT_GROUPS
from loader import load_survival, load_cluster, load_user_stat

st.set_page_config(page_title="메이플스토리 유저 분석", layout="wide")

SEGMENT_ORDER  = ['Lv.285~289', 'Lv.290~294', 'Lv.295~299']
CLUSTER_COLORS = px.colors.qualitative.Set2


# ── KM 계산 헬퍼 ─────────────────────────────────────────────────────────────

def _km_curve(durations, events):
    """최소한의 Kaplan-Meier 계산 (lifelines 없이 직접 구현)."""
    df = pd.DataFrame({'t': durations, 'e': events}).sort_values('t').reset_index(drop=True)
    times   = [0]
    survive = [1.0]
    ci_lo   = [1.0]
    ci_hi   = [1.0]

    n = len(df)
    S = 1.0
    prev_t = -1

    for t_val, grp in df.groupby('t'):
        d = grp['e'].sum()
        n_at_risk = (df['t'] >= t_val).sum()
        if n_at_risk == 0:
            continue
        if t_val != prev_t:
            S = S * (1 - d / n_at_risk)
            # Greenwood variance
            with np.errstate(divide='ignore', invalid='ignore'):
                var = S ** 2 * ((df[df['t'] <= t_val]['e'].sum() / max(1, n_at_risk)) * (d / max(1, n_at_risk * (n_at_risk - d))) if (n_at_risk - d) > 0 else 0)
            se  = np.sqrt(max(var, 0))
            times.append(t_val)
            survive.append(S)
            ci_lo.append(max(0, S - 1.96 * se))
            ci_hi.append(min(1, S + 1.96 * se))
            prev_t = t_val

    return pd.DataFrame({'time': times, 'survival': survive, 'ci_lo': ci_lo, 'ci_hi': ci_hi})


def make_km_figure(df_surv, group_col, title, colors=None):
    groups = sorted(df_surv[group_col].dropna().unique())
    if colors is None:
        colors = px.colors.qualitative.Set2

    fig = go.Figure()
    medians = []

    for i, g in enumerate(groups):
        sub  = df_surv[df_surv[group_col] == g]
        km   = _km_curve(sub['duration'].values, sub['event'].values)
        col  = colors[i % len(colors)]
        name = str(g)

        # 신뢰구간 영역
        fig.add_trace(go.Scatter(
            x=np.concatenate([km['time'], km['time'][::-1]]),
            y=np.concatenate([km['ci_hi'], km['ci_lo'][::-1]]),
            fill='toself', fillcolor=col, opacity=0.15,
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        # KM 곡선
        fig.add_trace(go.Scatter(
            x=km['time'], y=km['survival'],
            mode='lines', name=name, line=dict(color=col, width=2),
        ))

        # 중앙 이탈 시점 (S <= 0.5 첫 지점)
        med_row = km[km['survival'] <= 0.5]
        med     = med_row['time'].iloc[0] if not med_row.empty else None
        medians.append({'그룹': name, '중앙 이탈 시점 (일)': med if med is not None else 'N/A (미도달)', '유저 수': len(sub)})

    fig.add_hline(y=0.5, line_dash='dot', line_color='gray',
                  annotation_text='50% 이탈선', annotation_position='right')
    fig.update_layout(
        title=title, xaxis_title='관찰 일수', yaxis_title='생존율',
        yaxis=dict(range=[0, 1.05]), legend_title=group_col,
    )
    return fig, pd.DataFrame(medians)


# ── 클러스터 레이블 ───────────────────────────────────────────────────────────

def label_cluster(df_cl):
    """active_day_ratio + avg_exp_pct 기준 간단 레이블링."""
    def _label(row):
        r, p = row['active_day_ratio'], row['avg_exp_pct']
        if r >= 0.6 and p >= 60:  return '고활동·고효율'
        if r >= 0.6 and p <  60:  return '고활동·저효율'
        if r <  0.6 and p >= 60:  return '저활동·고효율'
        return '저활동·저효율'
    df_cl = df_cl.copy()
    df_cl['cluster_label'] = df_cl.apply(_label, axis=1)
    return df_cl


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    st.title("메이플스토리 유저 분석")
    st.caption("유저 이탈 패턴 분석 · 클러스터별 특성 분석 · 스펙 분포 비교")
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "유저 이탈 분석",
        "클러스터 분석",
        "스펙 분포 분석",
    ])

    # ── TAB 1: Kaplan-Meier ──────────────────────────────────────────────────
    with tab1:
        df_surv = load_survival()
        if df_surv is None:
            st.warning(f"생존 분석 데이터가 없습니다.\n\n경로: `{SURVIVAL_FILE}`")
        else:
            st.markdown(
                "**7일 연속 비활성**을 이탈로 정의합니다 (Lv.285+ 고레벨 유저 기준).  \n"
                "Kaplan-Meier 곡선으로 레벨 구간별 · 클러스터별 이탈 시점을 비교합니다."
            )
            total = len(df_surv)
            churned = df_surv['event'].sum()
            st.caption(f"전체 유저: **{total:,}명**  |  이탈 확정: **{int(churned):,}명** ({churned/total*100:.1f}%)")

            c_grp = st.radio("그룹 기준:", ["레벨 구간 (segment)", "클러스터 (cluster)"],
                              horizontal=True)

            if c_grp == "레벨 구간 (segment)":
                df_plot = df_surv[df_surv['segment'].isin(SEGMENT_ORDER)].copy()
                fig, med_df = make_km_figure(df_plot, 'segment', 'Kaplan-Meier: 레벨 구간별 이탈 곡선')
            else:
                if 'cluster' not in df_surv.columns:
                    st.info("생존 데이터에 cluster 컬럼이 없습니다.")
                    return
                fig, med_df = make_km_figure(df_surv.dropna(subset=['cluster']),
                                             'cluster', 'Kaplan-Meier: 클러스터별 이탈 곡선',
                                             colors=CLUSTER_COLORS)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("중앙 이탈 시점 (Median Survival)")
            st.dataframe(med_df.set_index('그룹'), use_container_width=True)

            st.divider()
            st.subheader("이탈/잔류 비율 (레벨 구간별)")
            seg_cnt = df_surv.groupby(['segment', 'event']).size().unstack(fill_value=0)
            seg_cnt.columns = ['잔류', '이탈'] if 0 in seg_cnt.columns else seg_cnt.columns
            seg_pct = seg_cnt.div(seg_cnt.sum(axis=1), axis=0) * 100
            fig_bar = px.bar(seg_pct.reset_index(), x='segment', y=['잔류', '이탈'],
                             title='레벨 구간별 이탈/잔류 비율',
                             labels={'value': '비율 (%)', 'segment': '레벨 구간', 'variable': ''},
                             barmode='stack', color_discrete_sequence=['#4C9BE8', '#E8604C'])
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── TAB 2: 클러스터 분석 ─────────────────────────────────────────────────
    with tab2:
        df_cl = load_cluster()
        if df_cl is None:
            st.warning(f"클러스터 데이터가 없습니다.\n\n경로: `{CLUSTER_FILE}`")
        else:
            df_cl = label_cluster(df_cl)

            st.markdown("K-Means 클러스터링 결과 (k=4, 실루엣 0.3193)를 기반으로 유저 그룹을 분석합니다.")

            # 클러스터 요약
            profile_cols = ['active_day_ratio', 'avg_exp_pct', 'union_level', 'character_age_days']
            existing_cols = [c for c in profile_cols if c in df_cl.columns]
            profile = df_cl.groupby('cluster')[existing_cols].agg(['mean', 'median', 'count'])
            profile.columns = ['_'.join(c) for c in profile.columns]

            col_a, col_b = st.columns([1, 2])

            with col_a:
                st.subheader("클러스터 레이블")
                label_df = df_cl.groupby('cluster')['cluster_label'].first().reset_index()
                cnt_df   = df_cl.groupby('cluster').size().reset_index(name='유저 수')
                st.dataframe(label_df.merge(cnt_df, on='cluster').set_index('cluster'),
                             use_container_width=True)

            with col_b:
                st.subheader("주요 특성 비교 (클러스터별 중앙값)")
                show_cols = {
                    'active_day_ratio_median': '활동 일수 비율',
                    'avg_exp_pct_median': '평균 경험치 퍼센타일',
                    'union_level_median': '유니온 레벨',
                }
                show_df = profile[[c for c in show_cols if c in profile.columns]].copy()
                show_df.columns = [show_cols[c] for c in show_df.columns]
                st.dataframe(show_df.round(2), use_container_width=True)

            st.divider()
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("세그먼트 × 클러스터 분포")
                if 'segment' in df_cl.columns:
                    cross = df_cl.groupby(['segment', 'cluster']).size().reset_index(name='count')
                    fig_cross = px.bar(cross, x='segment', y='count', color='cluster',
                                       barmode='stack', title='세그먼트별 클러스터 구성',
                                       labels={'count': '유저 수', 'segment': '레벨 구간', 'cluster': '클러스터'},
                                       color_discrete_sequence=CLUSTER_COLORS)
                    st.plotly_chart(fig_cross, use_container_width=True)

            with c2:
                st.subheader("활동 비율 vs 경험치 퍼센타일")
                if {'active_day_ratio', 'avg_exp_pct'}.issubset(df_cl.columns):
                    sample = df_cl.sample(min(1000, len(df_cl)), random_state=42)
                    fig_sc = px.scatter(sample, x='active_day_ratio', y='avg_exp_pct',
                                       color='cluster', opacity=0.6,
                                       title='클러스터별 활동 분포',
                                       labels={'active_day_ratio': '활동 일수 비율',
                                               'avg_exp_pct': '평균 경험치 퍼센타일',
                                               'cluster': '클러스터'},
                                       color_discrete_sequence=CLUSTER_COLORS)
                    st.plotly_chart(fig_sc, use_container_width=True)

            # Pre/Post 평균 비교
            if {'pre_avg', 'post_avg'}.issubset(df_cl.columns):
                st.divider()
                st.subheader("클러스터별 쇼케이스 전·후 평균 경험치")
                pp = df_cl.groupby('cluster')[['pre_avg', 'post_avg']].median().reset_index()
                pp_melt = pp.melt(id_vars='cluster', var_name='구간', value_name='중앙 경험치')
                pp_melt['구간'] = pp_melt['구간'].map({'pre_avg': '쇼케이스 전', 'post_avg': '쇼케이스 후'})
                fig_pp = px.bar(pp_melt, x='cluster', y='중앙 경험치', color='구간',
                                barmode='group', title='클러스터별 Pre/Post 중앙 경험치',
                                color_discrete_sequence=['#4C9BE8', '#E8604C'])
                st.plotly_chart(fig_pp, use_container_width=True)

    # ── TAB 3: 스펙 분포 분석 ────────────────────────────────────────────────
    with tab3:
        df_stat = load_user_stat()
        if df_stat is None:
            st.warning("유저 스펙 데이터가 없습니다.")
        else:
            st.markdown("유저 스펙 데이터 기반으로 레벨 구간 · 클러스터별 능력치 분포를 분석합니다.")

            group_by = st.radio("그룹 기준:", ["레벨 구간 (tier)", "월드 그룹 (world_group)"],
                                 horizontal=True, key='stat_group_by')
            group_col = 'tier' if '(tier)' in group_by else 'world_group'

            stat_group = st.selectbox("스탯 그룹 선택:", list(STAT_GROUPS.keys()))
            stat_cols  = [c for c in STAT_GROUPS[stat_group] if c in df_stat.columns]

            if not stat_cols:
                st.info(f"'{stat_group}' 그룹에 해당하는 스탯 컬럼이 데이터에 없습니다.")
            else:
                sel_stat = st.selectbox("스탯 선택:", stat_cols)
                sub = df_stat[[group_col, sel_stat]].dropna()

                if sub.empty:
                    st.info("표시할 데이터가 없습니다.")
                else:
                    col_v, col_m = st.columns([3, 2])

                    with col_v:
                        fig_vio = px.violin(sub, x=group_col, y=sel_stat, box=True, points=False,
                                            title=f"{sel_stat} — {group_col}별 분포",
                                            labels={group_col: group_col, sel_stat: sel_stat},
                                            color=group_col)
                        st.plotly_chart(fig_vio, use_container_width=True)

                    with col_m:
                        st.subheader("그룹별 통계 요약")
                        desc = sub.groupby(group_col)[sel_stat].describe()[['count', 'mean', '50%', 'std']]
                        desc.columns = ['유저 수', '평균', '중앙값', '표준편차']
                        st.dataframe(desc.round(2), use_container_width=True)

                st.divider()
                st.subheader("전투력 중앙값 히트맵 (tier × world_group)")
                power_col = '전투력'
                if power_col in df_stat.columns and 'tier' in df_stat.columns and 'world_group' in df_stat.columns:
                    pivot_hp = df_stat.pivot_table(
                        values=power_col, index='tier', columns='world_group',
                        aggfunc='median'
                    )
                    if not pivot_hp.empty:
                        fig_hp = px.imshow(pivot_hp, text_auto='.3s',
                                           color_continuous_scale='Blues',
                                           title='tier × world_group 전투력 중앙값',
                                           labels={'color': '중앙값', 'x': 'world_group', 'y': 'tier'})
                        st.plotly_chart(fig_hp, use_container_width=True)
                    else:
                        st.info("전투력 히트맵 데이터가 부족합니다.")
                else:
                    st.info("전투력 컬럼 또는 tier/world_group 컬럼이 없습니다.")


if __name__ == "__main__":
    main()
