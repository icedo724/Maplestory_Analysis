import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ================= CONFIG =================
st.set_page_config(page_title="메이플 쇼케이스 영향 분석", page_icon="📊", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.dirname(os.path.dirname(current_dir))
AGG_DIR     = os.path.join(base_dir, "data", "showcase", "aggregated")

SHOWCASE_DATE = "2025-12-13"
# ==========================================


@st.cache_data
def load_agg():
    def p(name): return os.path.join(AGG_DIR, name)
    daily   = pd.read_csv(p('agg_daily_segment.csv'),   parse_dates=['date'])
    summary = pd.read_csv(p('agg_segment_summary.csv'))
    sun_ev  = pd.read_csv(p('agg_sunday_events.csv'),   parse_dates=['Date'])
    sun_box = pd.read_csv(p('agg_sunday_box.csv'))
    anova   = pd.read_csv(p('agg_anova.csv'))
    weekday = pd.read_csv(p('agg_weekday.csv'))
    job     = pd.read_csv(p('agg_job_summary.csv'))
    tukey_path = p('agg_tukey.csv')
    tukey   = pd.read_csv(tukey_path) if os.path.exists(tukey_path) else pd.DataFrame()
    return daily, summary, sun_ev, sun_box, anova, weekday, job, tukey


def main():
    if not os.path.exists(AGG_DIR):
        st.error(f"집계 데이터 폴더가 없습니다: `{AGG_DIR}`  \n"
                 "E_export_agg.py를 먼저 실행해 주세요.")
        return

    daily, summary, sun_ev, sun_box, anova, weekday, job, tukey = load_agg()

    # 메타 정보
    sym_days   = int(summary['sym_days'].iloc[0])
    data_start = summary['data_start'].iloc[0]
    data_end   = summary['data_end'].iloc[0]
    total_n    = int(summary['n'].sum())
    showcase_dt = pd.to_datetime(SHOWCASE_DATE)
    sym_start  = showcase_dt - pd.Timedelta(days=sym_days)
    sym_end    = showcase_dt + pd.Timedelta(days=sym_days)

    st.title("📊 메이플스토리 쇼케이스 영향 분석 대시보드")
    st.caption(f"쇼케이스: {SHOWCASE_DATE}  |  분석 대상: {total_n:,}명  |"
               f"  수집 기간: {data_start} ~ {data_end}")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 쇼케이스 영향 (Pre vs Post)",
        "📅 주간 패턴 (메요일/선데이)",
        "☀️ 선데이 이벤트 심층 분석",
        "🔍 직업군 반응 비교",
    ])

    # ──────────────────────────────────────────
    # TAB 1: 쇼케이스 영향 분석
    # ──────────────────────────────────────────
    with tab1:
        st.markdown("쇼케이스 전후 **동일한 기간**을 기준으로 레벨링 동기 변화를 분석합니다.")
        st.info(f"**대칭 분석 기간:** {sym_start.strftime('%Y-%m-%d')} ~ {sym_end.strftime('%Y-%m-%d')}"
                f"  (쇼케이스 기준 ±{sym_days}일)  |  전체 데이터: {data_start} ~ {data_end}")

        df_sym = daily[(daily['date'] >= sym_start) & (daily['date'] <= sym_end)]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("전체 유저 평균 성장 추이")
            trend_total = df_sym.groupby('date')['avg_exp'].mean().reset_index()
            fig_total = px.line(trend_total, x='date', y='avg_exp', markers=True,
                                labels={'avg_exp': '평균 일일 경험치', 'date': '날짜'})
            fig_total.add_vline(x=SHOWCASE_DATE, line_width=2, line_dash="dash", line_color="red")
            fig_total.add_annotation(x=SHOWCASE_DATE, y=1, yref="paper",
                                     text="Showcase", showarrow=False, font=dict(color="red", size=13))
            st.plotly_chart(fig_total, use_container_width=True)

        with col2:
            st.subheader("레벨 구간별 평상시 대비 변화율 (%)")
            baseline = summary.set_index('segment')['pre_avg'].to_dict()
            df_sym2  = df_sym.copy()
            df_sym2['baseline'] = df_sym2['segment'].map(baseline)
            df_sym2['Exp_Ratio'] = df_sym2['avg_exp'] / df_sym2['baseline'] * 100

            fig_seg = px.line(df_sym2, x='date', y='Exp_Ratio', color='segment', markers=True,
                              labels={'Exp_Ratio': '평상시 대비 (%)', 'date': '날짜'})
            fig_seg.add_hline(y=100, line_dash="dot", line_color="gray")
            fig_seg.add_vline(x=SHOWCASE_DATE, line_width=2, line_dash="dash", line_color="red")
            fig_seg.add_annotation(x=SHOWCASE_DATE, y=1, yref="paper",
                                   text="Showcase", showarrow=False, font=dict(color="red", size=13))
            st.plotly_chart(fig_seg, use_container_width=True)

        st.divider()
        st.subheader("🧪 통계 검증: 쇼케이스 전/후 유의미한 변화가 있었는가?")
        st.caption(f"대칭 기간 ±{sym_days}일 기준")

        c_bar, c_stat = st.columns([2, 1.5])

        with c_bar:
            fig_bar = px.bar(summary, x='segment', y='growth_rate', color='segment',
                             title=f"구간별 성장 증감률 (%) — 대칭 ±{sym_days}일 기준",
                             text='growth_rate',
                             labels={'growth_rate': '증감률 (%)', 'segment': '구간'})
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_stat:
            st.markdown("#### 대응표본 t-검정 결과")
            for _, row in summary.iterrows():
                p = row['p_value']
                is_inc   = row['post_avg'] > row['pre_avg']
                dir_icon = "📈 증가" if is_inc else "📉 감소"
                sig = ("매우 유의미 (⭐⭐⭐)" if p < 0.001
                       else "유의미 (⭐)" if p < 0.05
                       else "차이 없음 (❌)")
                st.info(f"**[{row['segment']}]** n={int(row['n']):,}\n"
                        f"* 변화: {dir_icon} ({sig})\n"
                        f"* P-value: {p:.4e}")

    # ──────────────────────────────────────────
    # TAB 2: 메요일 / 주간 패턴
    # ──────────────────────────────────────────
    with tab2:
        st.markdown("수요일 시작 7일 단위로 요일별 사냥 패턴을 분석합니다.")
        day_order = ['수', '목', '금', '토', '일', '월', '화']

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            all_weeks = sorted(weekday['week_idx'].unique())
            opts_w    = ["전체 주차 (평균)"] + [f"{w}주차" for w in all_weeks]
            sel_w     = st.selectbox("📅 주차 선택:", opts_w)
        with col_s2:
            opts_s = ["전체 유저"] + sorted(weekday['segment'].dropna().unique().tolist())
            sel_s  = st.selectbox("📊 그룹 선택:", opts_s)

        f_df = weekday.copy()
        if sel_w != "전체 주차 (평균)":
            f_df = f_df[f_df['week_idx'] == int(sel_w.replace("주차", ""))]
        if sel_s != "전체 유저":
            f_df = f_df[f_df['segment'] == sel_s]

        t_weekly = f_df.groupby('day_name')['avg_exp'].mean().reindex(day_order).reset_index()
        t_weekly.columns = ['day_name', 'avg_exp']

        fig_week = px.line(t_weekly, x='day_name', y='avg_exp', markers=True,
                           title=f"[{sel_w}] {sel_s} 요일별 평균 경험치",
                           labels={'avg_exp': '평균 일일 경험치', 'day_name': '요일'})
        fig_week.add_vrect(x0=-0.3, x1=2.3, fillcolor="LightSteelBlue",
                           opacity=0.2, layer="below", line_width=0)
        y_m = t_weekly['avg_exp'].max() if not t_weekly.empty else 1
        fig_week.add_vline(x=1, line_width=1.5, line_dash="dot", line_color="orange")
        fig_week.add_annotation(x=1, y=y_m * 0.1, text="목(메요일)",
                                showarrow=False, font=dict(color="orange"), textangle=-90)
        fig_week.add_vline(x=4, line_width=1.5, line_dash="dot", line_color="green")
        fig_week.add_annotation(x=4, y=y_m * 0.1, text="일(선데이)",
                                showarrow=False, font=dict(color="green"), textangle=-90)
        st.plotly_chart(fig_week, use_container_width=True)

        st.subheader("주차 × 요일 히트맵")
        hm_df = weekday.copy()
        if sel_s != "전체 유저":
            hm_df = hm_df[hm_df['segment'] == sel_s]
        pivot = hm_df.pivot_table(index='week_idx', columns='day_name',
                                  values='avg_exp', aggfunc='mean')
        pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
        pivot.index = [f"{i}주차" for i in pivot.index]

        fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues",
                             labels={'color': '평균 경험치', 'x': '요일', 'y': '주차'},
                             title=f"{sel_s} 주차별 요일 경험치 히트맵")
        showcase_week_idx = int(((showcase_dt - pd.Timedelta(days=2) -
                                  (pd.to_datetime(daily['date'].min()) - pd.Timedelta(days=2))).days) // 7) + 1
        sc_label = f"{showcase_week_idx}주차"
        if sc_label in pivot.index:
            sc_y = list(pivot.index).index(sc_label)
            fig_heat.add_hline(y=sc_y, line_dash="dash", line_color="red", line_width=1.5,
                               annotation_text="쇼케이스", annotation_position="right")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ──────────────────────────────────────────
    # TAB 3: 선데이 메이플 이벤트 심층 분석
    # ──────────────────────────────────────────
    with tab3:
        st.markdown(
            "선데이 메이플 이벤트 종류(경타포스 / 사냥 / 사냥 외)에 따라 "
            "유저들의 **사냥 동기 변화율**이 달라지는지 검증합니다.  \n"
            "절댓값 경험치가 아닌 **유저별 평상시(Pre_Avg) 대비 해당 선데이의 변화율**을 "
            "기준으로 분석하여 레벨 구간 간 기본 경험치 차이를 제거합니다."
        )

        cat_order = ['경타포스', '사냥', '사냥 외']

        col_sub1, col_sub2 = st.columns([1, 2.5])

        with col_sub1:
            st.info("📌 선데이 분류 현황")
            log_display = (sun_ev[['Date', 'Sunday_Type', 'Event_Category']]
                           .drop_duplicates()
                           .sort_values('Date'))
            log_display['Date'] = log_display['Date'].dt.strftime('%Y-%m-%d')
            log_display['Sunday_Type'] = log_display['Sunday_Type'].fillna('기록없음(일반)')
            st.dataframe(log_display.reset_index(drop=True), use_container_width=True)

            sel_seg_sun = st.selectbox(
                "📊 분석 그룹:",
                ["전체 유저"] + sorted(sun_box['scope'].unique().tolist()),
                key='sun_seg'
            )

        with col_sub2:
            scope_key = "전체" if sel_seg_sun == "전체 유저" else sel_seg_sun
            box_data  = sun_box[sun_box['scope'] == scope_key]

            fig_box = go.Figure()
            colors  = px.colors.qualitative.Plotly
            for i, cat in enumerate(cat_order):
                row = box_data[box_data['category'] == cat]
                if row.empty:
                    continue
                r = row.iloc[0]
                fig_box.add_trace(go.Box(
                    name=cat,
                    q1=[r['p25']], median=[r['median']], q3=[r['p75']],
                    lowerfence=[r['p5']], upperfence=[r['p95']],
                    mean=[r['mean']],
                    boxmean=True,
                    marker_color=colors[i],
                    x=[cat],
                ))
            fig_box.add_hline(y=100, line_dash="dot", line_color="gray",
                              annotation_text="평상시 기준(100%)", annotation_position="right")
            fig_box.update_layout(
                title=f"[{sel_seg_sun}] 이벤트 유형별 평상시 대비 사냥량 변화율 (상위 1% 제외)",
                yaxis_title='평상시 대비 (%)', xaxis_title='이벤트 분류',
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # 카테고리별 평균 변화율 막대차트
        st.subheader("이벤트 유형별 평균 변화율 비교")
        box_all = sun_box[sun_box['scope'] == scope_key].copy()
        box_all['category'] = pd.Categorical(box_all['category'], categories=cat_order, ordered=True)
        box_all = box_all.sort_values('category')

        fig_mean = px.bar(box_all, x='category', y='mean', color='category',
                          text='mean',
                          category_orders={'category': cat_order},
                          labels={'category': '이벤트 분류', 'mean': '평균 변화율 (%)'},
                          title="이벤트 유형별 평상시 대비 평균 사냥량 (%)")
        fig_mean.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_mean.add_hline(y=100, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_mean, use_container_width=True)

        n_info = "  |  ".join([f"**{r['category']}** n={int(r['n']):,}"
                               for _, r in box_all.iterrows()])
        st.caption(f"그룹별 (날짜 × 유저) 샘플 수: {n_info}")

        st.divider()
        st.subheader("🧪 ANOVA 검정: 이벤트 유형에 따른 사냥량 변화율 차이")
        st.caption("귀무가설: 모든 이벤트 유형에서 평상시 대비 사냥량 변화율이 동일하다.")

        anova_row = anova[anova['scope'] == scope_key]
        if not anova_row.empty:
            r       = anova_row.iloc[0]
            f_stat  = r['f_stat']
            p_val   = r['p_value']
            mean_cols = {c.replace('mean_', ''): r[c]
                         for c in r.index if c.startswith('mean_')}
            means_text = " | ".join([f"**{k}**: {v:.1f}%" for k, v in mean_cols.items()])

            if p_val < 0.05:
                st.success(
                    f"**이벤트 유형 간 사냥량 변화율에 유의미한 차이 존재 "
                    f"(F={f_stat:.2f}, P={p_val:.4e})**\n\n"
                    f"선데이 이벤트의 종류가 유저 사냥 동기에 실질적으로 다른 영향을 미쳤습니다.\n\n"
                    f"*(평균 변화율: {means_text})*"
                )
            else:
                st.info(
                    f"**이벤트 유형 간 통계적으로 유의미한 차이 없음 "
                    f"(F={f_stat:.2f}, P={p_val:.4f})**\n\n"
                    f"*(평균 변화율: {means_text})*"
                )

            if not tukey.empty:
                st.markdown("#### 사후 검정 (Tukey HSD): 어떤 그룹 간 차이가 있는가?")
                tukey_scope = tukey[tukey['scope'] == scope_key].copy()
                tukey_scope['유의미'] = tukey_scope['reject'].apply(
                    lambda x: "✅ 차이 있음" if x else "❌ 차이 없음"
                )
                st.dataframe(
                    tukey_scope[['group1', 'group2', 'meandiff', 'p-adj', '유의미']],
                    use_container_width=True
                )

    # ──────────────────────────────────────────
    # TAB 4: 직업군 반응 비교
    # ──────────────────────────────────────────
    with tab4:
        st.markdown(
            "쇼케이스에서 발표된 **특정 직업군 유저**와 나머지 유저의 반응 차이를 비교합니다.  \n"
            "쇼케이스 콘텐츠가 자신의 직업과 관련될 때 더 강하게 레벨링 동기가 올라가는지 확인할 수 있습니다."
        )

        job_list = sorted(job['job'].dropna().unique().tolist())

        col_j1, col_j2 = st.columns([1, 2])
        with col_j1:
            st.markdown("#### 발표 직업군 선택")
            selected_jobs = st.multiselect("직업 선택 (복수 가능):", options=job_list,
                                           default=[], key='job_select')
            sel_seg_job = st.selectbox("레벨 구간 필터:",
                                       ["전체"] + sorted(job['segment'].dropna().unique().tolist()),
                                       key='seg_job')

        with col_j2:
            if not selected_jobs:
                st.info("왼쪽에서 쇼케이스 발표 직업군을 선택하면 비교 분석이 시작됩니다.")

                # 선택 전에는 전체 직업 변화율 순위 표시
                st.subheader("전체 직업별 성장 변화율 순위")
                job_disp = job.copy()
                if sel_seg_job != "전체":
                    job_disp = job_disp[job_disp['segment'] == sel_seg_job]
                job_disp = job_disp[job_disp['n'] >= 5].sort_values('growth_rate', ascending=True)
                fig_all = px.bar(job_disp.tail(20), x='growth_rate', y='job',
                                 orientation='h', color='growth_rate',
                                 color_continuous_scale='RdYlGn',
                                 title="상위 20개 직업 성장 변화율 (%)",
                                 labels={'growth_rate': '변화율 (%)', 'job': '직업'})
                st.plotly_chart(fig_all, use_container_width=True)
            else:
                df_job = job.copy()
                if sel_seg_job != "전체":
                    df_job = df_job[df_job['segment'] == sel_seg_job]

                df_job['Group'] = df_job['job'].apply(
                    lambda j: '발표 직업군' if j in selected_jobs else '기타 직업군'
                )
                group_summary = (df_job.groupby('Group')
                                 .apply(lambda x: pd.Series({
                                     'n':          x['n'].sum(),
                                     'pre_avg':    np.average(x['pre_avg'],  weights=x['n']),
                                     'post_avg':   np.average(x['post_avg'], weights=x['n']),
                                 }))
                                 .reset_index())
                group_summary['growth_rate'] = ((group_summary['post_avg'] - group_summary['pre_avg'])
                                                / group_summary['pre_avg'] * 100)

                fig_job = go.Figure()
                for grp, color in [('발표 직업군', '#EF553B'), ('기타 직업군', '#636EFA')]:
                    row = group_summary[group_summary['Group'] == grp]
                    if row.empty:
                        continue
                    fig_job.add_trace(go.Bar(name=f'{grp} Pre',  x=[grp],
                                            y=row['pre_avg'].values,
                                            marker_color=color, opacity=0.5, legendgroup=grp))
                    fig_job.add_trace(go.Bar(name=f'{grp} Post', x=[grp],
                                            y=row['post_avg'].values,
                                            marker_color=color, opacity=1.0, legendgroup=grp))
                fig_job.update_layout(barmode='group',
                                      title='발표 직업군 vs 기타: Pre / Post 평균 경험치',
                                      yaxis_title='평균 일일 경험치')
                st.plotly_chart(fig_job, use_container_width=True)

                for _, row in group_summary.iterrows():
                    sig = "유의미 ✅" if abs(row['growth_rate']) > 5 else "미미한 차이"
                    st.metric(label=f"{row['Group']} (n={int(row['n']):,})",
                              value=f"{row['growth_rate']:+.1f}%",
                              delta=f"Pre: {row['pre_avg']:,.0f} → Post: {row['post_avg']:,.0f}")


if __name__ == "__main__":
    main()
