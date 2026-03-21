import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
import os

# ================= CONFIG =================
st.set_page_config(page_title="메이플 쇼케이스 영향 분석", page_icon="📊", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir    = os.path.dirname(os.path.dirname(current_dir))

PROCESSED_PATH  = os.path.join(base_dir, "data", "showcase", "preprocessed", "daily_segment_processed.csv")
SUNDAY_LOG_PATH = os.path.join(base_dir, "data", "sundaylog.txt")
SHOWCASE_DATE   = "2025-12-13"
MIN_VALID_DAYS  = 7   # Pre/Post 평균 신뢰도 최소 기준일
# ==========================================


@st.cache_data
def load_and_prepare():
    """데이터 로드 + 공통 전처리를 캐시로 묶어 탭 진입마다 재연산 방지"""
    if not os.path.exists(PROCESSED_PATH):
        return None, None

    df = pd.read_csv(PROCESSED_PATH)

    daily_cols = sorted(
        [c for c in df.columns if c.startswith('Daily_')],
        key=lambda x: pd.to_datetime(x.replace('Daily_', ''))
    )

    # Wide → Long 변환
    melted = df.melt(
        id_vars=['segment', 'name', 'job', 'world'],
        value_vars=daily_cols,
        var_name='Date_Col',
        value_name='Exp'
    )

    # Daily_YYYY-MM-DD = 전날→당일 증분.
    # 실제 사냥 시점은 전날이므로 -1일 보정 (의도된 보정)
    melted['Date']      = pd.to_datetime(melted['Date_Col'].str.replace('Daily_', '')) - pd.Timedelta(days=1)
    melted['DayOfWeek'] = melted['Date'].dt.dayofweek  # ← 탭 밖에서 계산 (Tab 3 의존성 해소)

    # 주차 인덱스 (수요일 시작)
    shifted_date        = melted['Date'] - pd.Timedelta(days=2)
    min_shifted         = shifted_date.min()
    melted['Week_Idx']  = ((shifted_date - min_shifted).dt.days // 7) + 1
    melted['Week_Label'] = melted['Week_Idx'].astype(str) + "주차"

    day_map = {2: '수', 3: '목', 4: '금', 5: '토', 6: '일', 0: '월', 1: '화'}
    melted['Day_Name'] = melted['DayOfWeek'].map(day_map)

    return df, melted


@st.cache_data
def load_sunday_log():
    if not os.path.exists(SUNDAY_LOG_PATH):
        return pd.DataFrame(columns=['Date', 'Sunday_Type'])
    data = []
    with open(SUNDAY_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                date_str, event_type = line.strip().split(':', 1)
                data.append({'Date': pd.to_datetime(date_str.strip()), 'Sunday_Type': event_type.strip()})
    return pd.DataFrame(data)


def classify_sunday_event(event_str):
    if pd.isna(event_str):
        return '기타'
    event_str = str(event_str)
    if '경타포스' in event_str:
        return '경타포스'
    elif any(k in event_str for k in ['몬파', '룬콤보', '트레져', '솔에르다', '사냥']):
        return '사냥'
    elif any(k in event_str for k in ['강화', '샤타포스', '미라클']):
        return '강화'
    return '기타'


def main():
    df, melted = load_and_prepare()
    if df is None:
        st.error(f"데이터 파일이 없습니다: {PROCESSED_PATH}")
        return

    st.title("📊 메이플스토리 쇼케이스 영향 분석 대시보드")
    st.caption(f"쇼케이스 일자: {SHOWCASE_DATE}  |  분석 대상: {len(df):,}명")
    st.divider()

    showcase_dt = pd.to_datetime(SHOWCASE_DATE)

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

        # [수정] Pre 기간과 Post 기간 길이를 실제 데이터 기준으로 산출
        # min_date는 -1일 보정된 날짜 기준
        min_date = melted['Date'].min()
        max_date = melted['Date'].max()
        pre_days  = (showcase_dt - min_date).days
        post_days = (max_date - showcase_dt).days
        sym_days  = min(pre_days, post_days)  # 대칭 비교 가능한 최대 일수
        sym_start = showcase_dt - pd.Timedelta(days=sym_days)
        sym_end   = showcase_dt + pd.Timedelta(days=sym_days)

        st.info(
            f"**대칭 분석 기간:** {sym_start.strftime('%Y-%m-%d')} ~ {sym_end.strftime('%Y-%m-%d')} "
            f"(쇼케이스 기준 ±{sym_days}일)  |  "
            f"전체 데이터 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"
        )

        df_sym = melted[(melted['Date'] >= sym_start) & (melted['Date'] <= sym_end)]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("전체 유저 평균 성장 추이")
            trend_total = df_sym.groupby('Date')['Exp'].mean().reset_index()
            fig_total = px.line(
                trend_total, x='Date', y='Exp', markers=True,
                labels={'Exp': '평균 일일 경험치', 'Date': '날짜'}
            )
            fig_total.add_vline(
                x=SHOWCASE_DATE, line_width=2, line_dash="dash", line_color="red"
            )
            fig_total.add_annotation(
                x=SHOWCASE_DATE, y=1, yref="paper",
                text="Showcase", showarrow=False, font=dict(color="red", size=13)
            )
            st.plotly_chart(fig_total, use_container_width=True)

        with col2:
            st.subheader("레벨 구간별 평상시 대비 변화율 (%)")
            trend_seg  = df_sym.groupby(['segment', 'Date'])['Exp'].mean().reset_index()
            baseline   = df.groupby('segment')['Pre_Avg'].mean().reset_index()
            baseline.rename(columns={'Pre_Avg': 'Baseline_Exp'}, inplace=True)
            trend_seg  = pd.merge(trend_seg, baseline, on='segment')
            trend_seg['Exp_Ratio'] = (trend_seg['Exp'] / trend_seg['Baseline_Exp']) * 100

            fig_seg = px.line(
                trend_seg, x='Date', y='Exp_Ratio', color='segment', markers=True,
                labels={'Exp_Ratio': '평상시 대비 (%)', 'Date': '날짜'}
            )
            fig_seg.add_hline(y=100, line_dash="dot", line_color="gray")
            fig_seg.add_vline(
                x=SHOWCASE_DATE, line_width=2, line_dash="dash", line_color="red"
            )
            fig_seg.add_annotation(
                x=SHOWCASE_DATE, y=1, yref="paper",
                text="Showcase", showarrow=False, font=dict(color="red", size=13)
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        st.divider()
        st.subheader("🧪 통계 검증: 쇼케이스 전/후 유의미한 변화가 있었는가?")

        # [수정] MIN_VALID_DAYS 필터 적용 후 t-검정
        df_stat = df[
            (df['Pre_Valid_Days']  >= MIN_VALID_DAYS) &
            (df['Post_Valid_Days'] >= MIN_VALID_DAYS)
        ]
        excluded = len(df) - len(df_stat)
        if excluded > 0:
            st.caption(f"*유효 데이터 {MIN_VALID_DAYS}일 미만 유저 {excluded:,}명 검정 제외*")

        c_bar, c_stat = st.columns([2, 1.5])
        with c_bar:
            summary = df_stat.groupby('segment')[['Pre_Avg', 'Post_Avg']].mean().reset_index()
            summary['Growth_Rate'] = (summary['Post_Avg'] - summary['Pre_Avg']) / summary['Pre_Avg'] * 100
            fig_bar = px.bar(
                summary, x='segment', y='Growth_Rate', color='segment',
                title="구간별 성장 증감률 (%)", text_auto='.1f'
            )
            fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_stat:
            st.markdown("#### 대응표본 t-검정 결과")
            for seg in sorted(df_stat['segment'].dropna().unique()):
                seg_data = df_stat[df_stat['segment'] == seg].dropna(subset=['Pre_Avg', 'Post_Avg'])
                if len(seg_data) > 1:
                    t_stat, p_val = stats.ttest_rel(seg_data['Pre_Avg'], seg_data['Post_Avg'])
                    is_inc  = seg_data['Post_Avg'].mean() > seg_data['Pre_Avg'].mean()
                    dir_icon = "📈 증가" if is_inc else "📉 감소"
                    if p_val < 0.001:
                        sig = "매우 유의미 (⭐⭐⭐)"
                    elif p_val < 0.05:
                        sig = "유의미 (⭐)"
                    else:
                        sig = "차이 없음 (❌)"
                    st.info(
                        f"**[{seg}]** n={len(seg_data):,}\n"
                        f"* 변화: {dir_icon} ({sig})\n"
                        f"* P-value: {p_val:.4e}"
                    )

    # ──────────────────────────────────────────
    # TAB 2: 메요일 / 주간 패턴
    # ──────────────────────────────────────────
    with tab2:
        st.markdown("수요일 시작 7일 단위로 요일별 사냥 패턴을 분석합니다.")
        day_order = ['수', '목', '금', '토', '일', '월', '화']

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            opts_w = ["전체 주차 (평균)"] + sorted(
                melted['Week_Label'].unique().tolist(),
                key=lambda x: int(x.replace("주차", ""))
            )
            sel_w = st.selectbox("📅 주차 선택:", opts_w)
        with col_s2:
            opts_s = ["전체 유저"] + sorted(melted['segment'].dropna().unique().tolist())
            sel_s  = st.selectbox("📊 그룹 선택:", opts_s)

        f_df = melted.copy()
        if sel_w != "전체 주차 (평균)":
            f_df = f_df[f_df['Week_Label'] == sel_w]
        if sel_s != "전체 유저":
            f_df = f_df[f_df['segment'] == sel_s]

        t_weekly = f_df.groupby('Day_Name')['Exp'].mean().reset_index()
        t_weekly['Day_Name'] = pd.Categorical(t_weekly['Day_Name'], categories=day_order, ordered=True)
        t_weekly = t_weekly.sort_values('Day_Name')

        fig_week = px.line(
            t_weekly, x='Day_Name', y='Exp', markers=True,
            title=f"[{sel_w}] {sel_s} 요일별 평균 경험치",
            labels={'Exp': '평균 일일 경험치', 'Day_Name': '요일'}
        )
        # 수~금 강조 영역 (메요일 포함 주 초)
        fig_week.add_vrect(x0=-0.3, x1=2.3, fillcolor="LightSteelBlue", opacity=0.2, layer="below", line_width=0)
        y_m = t_weekly['Exp'].max() if not t_weekly.empty else 1
        fig_week.add_vline(x=1, line_width=1.5, line_dash="dot", line_color="orange")
        fig_week.add_annotation(x=1, y=y_m * 0.1, text="목(메요일)", showarrow=False,
                                font=dict(color="orange"), textangle=-90)
        fig_week.add_vline(x=4, line_width=1.5, line_dash="dot", line_color="green")
        fig_week.add_annotation(x=4, y=y_m * 0.1, text="일(선데이)", showarrow=False,
                                font=dict(color="green"), textangle=-90)
        st.plotly_chart(fig_week, use_container_width=True)

        # 추가: 주차별 히트맵 (요일 × 주차)
        st.subheader("주차 × 요일 히트맵")
        heatmap_df = melted.copy()
        if sel_s != "전체 유저":
            heatmap_df = heatmap_df[heatmap_df['segment'] == sel_s]

        pivot = heatmap_df.pivot_table(index='Week_Label', columns='Day_Name', values='Exp', aggfunc='mean')
        # 요일 순서 정렬
        pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
        pivot = pivot.reindex(sorted(pivot.index, key=lambda x: int(x.replace("주차", ""))))

        fig_heat = px.imshow(
            pivot, aspect="auto", color_continuous_scale="Blues",
            labels={'color': '평균 경험치', 'x': '요일', 'y': '주차'},
            title=f"{sel_s} 주차별 요일 경험치 히트맵"
        )
        # 쇼케이스 주차 강조선
        showcase_week_idx = int(((showcase_dt - pd.Timedelta(days=2) - (melted['Date'] - pd.Timedelta(days=2)).min()).days) // 7) + 1
        sc_week_label = f"{showcase_week_idx}주차"
        if sc_week_label in pivot.index:
            sc_y = list(pivot.index).index(sc_week_label)
            fig_heat.add_hline(y=sc_y, line_dash="dash", line_color="red", line_width=1.5,
                               annotation_text="쇼케이스", annotation_position="right")
        st.plotly_chart(fig_heat, use_container_width=True)

    # ──────────────────────────────────────────
    # TAB 3: 선데이 메이플 이벤트 심층 분석
    # ──────────────────────────────────────────
    with tab3:
        st.markdown(
            "선데이 메이플 이벤트 종류(경타포스 / 사냥 / 강화 / 기타)에 따라 "
            "유저들의 **사냥 동기 변화율**이 달라지는지 검증합니다.  \n"
            "절댓값 경험치가 아닌 **유저별 평상시(Pre_Avg) 대비 해당 선데이의 변화율**을 "
            "기준으로 분석하여 레벨 구간 간 기본 경험치 차이를 제거합니다."
        )

        sunday_log_df = load_sunday_log()

        if sunday_log_df.empty:
            st.warning(
                "⚠️ `sundaylog.txt` 파일이 없거나 비어 있습니다. "
                "선데이 이벤트 분류 없이 일반 일요일로만 분석됩니다."
            )

        # 일요일 데이터 추출 및 이벤트 분류 병합
        df_sunday_only   = melted[melted['DayOfWeek'] == 6].copy()
        df_sunday_merged = pd.merge(df_sunday_only, sunday_log_df, on='Date', how='left')
        df_sunday_merged['Event_Category'] = df_sunday_merged['Sunday_Type'].apply(classify_sunday_event)

        # ── 핵심: 유저별 Pre_Avg 대비 변화율 계산 ──────────────────────────
        # 절댓값 경험치는 레벨 구간마다 단위가 달라 구간 간 직접 비교 불가.
        # Pre_Avg(평상시 일평균)를 기준으로 해당 선데이 경험치를 표준화.
        # Ratio = (Sunday_Exp / Pre_Avg) * 100
        # Ratio > 100: 평상시보다 더 사냥함 / < 100: 덜 사냥함
        pre_avg_map = df.set_index('name')['Pre_Avg']
        df_sunday_merged['Pre_Avg'] = df_sunday_merged['name'].map(pre_avg_map)
        valid_mask = (
            df_sunday_merged['Pre_Avg'].notna() &
            (df_sunday_merged['Pre_Avg'] > 0) &
            df_sunday_merged['Exp'].notna()
        )
        df_sunday_merged['Exp_Ratio'] = np.where(
            valid_mask,
            (df_sunday_merged['Exp'] / df_sunday_merged['Pre_Avg']) * 100,
            np.nan
        )

        display_log = (
            df_sunday_merged[['Date', 'Sunday_Type', 'Event_Category']]
            .drop_duplicates()
            .sort_values('Date')
        )
        display_log['Date']        = display_log['Date'].dt.strftime('%Y-%m-%d')
        display_log['Sunday_Type'] = display_log['Sunday_Type'].fillna('기록없음(일반)')

        col_sub1, col_sub2 = st.columns([1, 2.5])
        with col_sub1:
            st.info("📌 선데이 분류 현황")
            st.dataframe(display_log.reset_index(drop=True), use_container_width=True)
            sel_seg_sun = st.selectbox(
                "📊 분석 그룹:", ["전체 유저"] + sorted(df_sunday_merged['segment'].dropna().unique().tolist()),
                key='sun_seg'
            )

        with col_sub2:
            target_df = (
                df_sunday_merged if sel_seg_sun == "전체 유저"
                else df_sunday_merged[df_sunday_merged['segment'] == sel_seg_sun]
            )
            cat_order = ['경타포스', '사냥', '강화', '기타']

            # 변화율 기준 박스플롯 (상위 1% 이상치 제거)
            q99      = target_df['Exp_Ratio'].quantile(0.99)
            plot_df  = target_df[target_df['Exp_Ratio'] <= q99]

            fig_box = px.box(
                plot_df, x='Event_Category', y='Exp_Ratio', color='Event_Category',
                title=f"[{sel_seg_sun}] 이벤트 유형별 평상시 대비 사냥량 변화율 (상위 1% 제외)",
                category_orders={'Event_Category': cat_order},
                labels={'Event_Category': '이벤트 분류', 'Exp_Ratio': '평상시 대비 (%)'},
                points=False
            )
            fig_box.add_hline(y=100, line_dash="dot", line_color="gray",
                              annotation_text="평상시 기준(100%)", annotation_position="right")
            st.plotly_chart(fig_box, use_container_width=True)

        # 카테고리별 평균 변화율 막대차트
        st.subheader("이벤트 유형별 평균 변화율 비교")
        cat_means = (
            target_df.groupby('Event_Category')['Exp_Ratio']
            .agg(['mean', 'median', 'count'])
            .reset_index()
            .rename(columns={'mean': '평균(%)', 'median': '중앙값(%)', 'count': 'n'})
        )
        cat_means['Event_Category'] = pd.Categorical(cat_means['Event_Category'], categories=cat_order, ordered=True)
        cat_means = cat_means.sort_values('Event_Category')

        fig_mean = px.bar(
            cat_means, x='Event_Category', y='평균(%)', color='Event_Category',
            text='평균(%)', category_orders={'Event_Category': cat_order},
            labels={'Event_Category': '이벤트 분류'},
            title="이벤트 유형별 평상시 대비 평균 사냥량 (%)"
        )
        fig_mean.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_mean.add_hline(y=100, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_mean, use_container_width=True)

        # n 수 표시
        n_info = "  |  ".join([
            f"**{row['Event_Category']}** n={int(row['n']):,}" for _, row in cat_means.iterrows()
        ])
        st.caption(f"그룹별 (날짜 × 유저) 샘플 수: {n_info}")

        st.divider()
        st.subheader("🧪 ANOVA 검정: 이벤트 유형에 따른 사냥량 변화율 차이")
        st.caption("귀무가설: 모든 이벤트 유형에서 평상시 대비 사냥량 변화율이 동일하다.")

        groups = {}
        for cat in cat_order:
            s = target_df[target_df['Event_Category'] == cat]['Exp_Ratio'].dropna()
            if len(s) > 1:
                groups[cat] = s

        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups.values())
            means_text = " | ".join([f"**{k}**: {v.mean():.1f}%" for k, v in groups.items()])

            if p_val < 0.05:
                st.success(
                    f"**이벤트 유형 간 사냥량 변화율에 유의미한 차이 존재 (F={f_stat:.2f}, P={p_val:.4e})**\n\n"
                    f"선데이 이벤트의 종류가 유저 사냥 동기에 실질적으로 다른 영향을 미쳤습니다.\n\n"
                    f"*(평균 변화율: {means_text})*"
                )
            else:
                st.info(
                    f"**이벤트 유형 간 통계적으로 유의미한 차이 없음 (F={f_stat:.2f}, P={p_val:.4f})**\n\n"
                    f"이벤트 종류보다 일요일 자체의 효과(요일 보너스)가 더 지배적이거나, "
                    f"표본 수가 부족할 수 있습니다.\n\n"
                    f"*(평균 변화율: {means_text})*"
                )

            # 사후 검정: 그룹이 3개 이상이면 Tukey HSD 로 쌍별 비교
            if len(groups) >= 3:
                st.markdown("#### 사후 검정 (Tukey HSD): 어떤 그룹 간 차이가 있는가?")
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    all_vals  = np.concatenate(list(groups.values()))
                    all_labels = np.concatenate([[k] * len(v) for k, v in groups.items()])
                    tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
                    tukey_df = pd.DataFrame(
                        data=tukey._results_table.data[1:],
                        columns=tukey._results_table.data[0]
                    )
                    tukey_df['유의미'] = tukey_df['reject'].apply(lambda x: "✅ 차이 있음" if x else "❌ 차이 없음")
                    st.dataframe(
                        tukey_df[['group1', 'group2', 'meandiff', 'p-adj', '유의미']],
                        use_container_width=True
                    )
                except ImportError:
                    st.caption("사후 검정을 위해 `pip install statsmodels` 가 필요합니다.")
        else:
            st.warning("분석 가능한 그룹이 2개 미만입니다. sundaylog.txt에 더 많은 날짜 기록이 필요합니다.")

    # ──────────────────────────────────────────
    # TAB 4: 직업군 반응 비교 (추가 기능)
    # 목적: 쇼케이스에서 발표된 직업군(예: 아델, 일리움 등)의
    #       유저들이 다른 직업군에 비해 더 강하게 반응했는지 확인
    # ──────────────────────────────────────────
    with tab4:
        st.markdown(
            "쇼케이스에서 발표된 **특정 직업군 유저**와 나머지 유저의 반응 차이를 비교합니다.  \n"
            "쇼케이스 콘텐츠가 자신의 직업과 관련될 때 더 강하게 레벨링 동기가 올라가는지 확인할 수 있습니다."
        )

        # 직업 목록 추출
        job_list = sorted(melted['job'].dropna().unique().tolist())

        col_j1, col_j2 = st.columns([1, 2])
        with col_j1:
            st.markdown("#### 발표 직업군 선택")
            st.caption("쇼케이스에서 다뤄진 직업군을 선택하세요.")
            selected_jobs = st.multiselect(
                "직업 선택 (복수 가능):",
                options=job_list,
                default=[],
                key='job_select'
            )
            sel_seg_job = st.selectbox(
                "레벨 구간 필터:",
                ["전체"] + sorted(df['segment'].dropna().unique().tolist()),
                key='seg_job'
            )

        with col_j2:
            if not selected_jobs:
                st.info("왼쪽에서 쇼케이스 발표 직업군을 선택하면 비교 분석이 시작됩니다.")
            else:
                df_job = df.copy()
                if sel_seg_job != "전체":
                    df_job = df_job[df_job['segment'] == sel_seg_job]
                df_job = df_job[
                    (df_job['Pre_Valid_Days']  >= MIN_VALID_DAYS) &
                    (df_job['Post_Valid_Days'] >= MIN_VALID_DAYS)
                ]

                df_job['Group'] = df_job['job'].apply(
                    lambda j: '발표 직업군' if j in selected_jobs else '기타 직업군'
                )

                group_summary = df_job.groupby('Group')[['Pre_Avg', 'Post_Avg']].mean().reset_index()
                group_summary['Growth_Rate'] = (
                    (group_summary['Post_Avg'] - group_summary['Pre_Avg'])
                    / group_summary['Pre_Avg'] * 100
                )

                # Pre / Post 막대 비교
                fig_job = go.Figure()
                for grp, color in [('발표 직업군', '#EF553B'), ('기타 직업군', '#636EFA')]:
                    row = group_summary[group_summary['Group'] == grp]
                    if row.empty:
                        continue
                    fig_job.add_trace(go.Bar(
                        name=f'{grp} Pre',
                        x=[grp], y=row['Pre_Avg'].values,
                        marker_color=color, opacity=0.5,
                        legendgroup=grp
                    ))
                    fig_job.add_trace(go.Bar(
                        name=f'{grp} Post',
                        x=[grp], y=row['Post_Avg'].values,
                        marker_color=color, opacity=1.0,
                        legendgroup=grp
                    ))

                fig_job.update_layout(
                    barmode='group',
                    title='발표 직업군 vs 기타: Pre / Post 평균 경험치',
                    yaxis_title='평균 일일 경험치',
                    xaxis_title=''
                )
                st.plotly_chart(fig_job, use_container_width=True)

                # 통계 검정
                for grp in ['발표 직업군', '기타 직업군']:
                    gd = df_job[df_job['Group'] == grp].dropna(subset=['Pre_Avg', 'Post_Avg'])
                    if len(gd) > 1:
                        _, pv = stats.ttest_rel(gd['Pre_Avg'], gd['Post_Avg'])
                        rate = group_summary[group_summary['Group'] == grp]['Growth_Rate'].values
                        rate_str = f"{rate[0]:+.1f}%" if len(rate) > 0 else "N/A"
                        sig = "유의미 ✅" if pv < 0.05 else "비유의 ❌"
                        st.metric(
                            label=f"{grp} (n={len(gd):,})",
                            value=rate_str,
                            delta=f"p={pv:.3e} | {sig}"
                        )

                # 증감률 직접 비교
                st.divider()
                st.markdown("#### 두 그룹의 증감률 직접 비교")
                group_data = {
                    grp: df_job[df_job['Group'] == grp]['Post_Avg'].values -
                         df_job[df_job['Group'] == grp]['Pre_Avg'].values
                    for grp in ['발표 직업군', '기타 직업군']
                }
                group_data = {k: v for k, v in group_data.items() if len(v) > 1}
                if len(group_data) == 2:
                    t2, p2 = stats.ttest_ind(*group_data.values())
                    if p2 < 0.05:
                        st.success(
                            f"**두 그룹 간 반응 차이가 통계적으로 유의미합니다 (p={p2:.4e})**\n\n"
                            f"발표 직업군 유저가 쇼케이스에 더 강하게 반응했음을 시사합니다."
                        )
                    else:
                        st.info(
                            f"두 그룹 간 반응 차이가 통계적으로 유의미하지 않습니다 (p={p2:.4f}).\n\n"
                            f"직업군과 무관하게 쇼케이스 효과가 나타났거나, 효과가 없을 수 있습니다."
                        )


if __name__ == "__main__":
    main()