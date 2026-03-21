import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
import os

# ================= CONFIG =================
st.set_page_config(page_title="메이플 레벨별 성장 분석", page_icon="📊", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))

PROCESSED_PATH = os.path.join(base_dir, "data", "showcase", "preprocessed", "daily_segment_processed.csv")
SUNDAY_LOG_PATH = os.path.join(base_dir, "data", "sundaylog.txt")
SHOWCASE_DATE = "2025-12-13"


@st.cache_data
def load_data():
    if not os.path.exists(PROCESSED_PATH): return None
    return pd.read_csv(PROCESSED_PATH)


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
    """ 선데이 이벤트 종류를 4가지 대분류로 매핑 """
    if pd.isna(event_str):
        return '기타'

    event_str = str(event_str)
    if '경타포스' in event_str:
        return '경타포스'
    elif any(k in event_str for k in ['몬파', '룬콤보', '트레져', '솔에르다', '사냥']):
        return '사냥'
    elif any(k in event_str for k in ['강화', '샤타포스', '미라클']):
        return '강화'
    else:
        return '기타'


def main():
    df = load_data()
    if df is None:
        st.error(f"데이터 파일이 없습니다. 경로를 확인해주세요: {PROCESSED_PATH}")
        return

    st.title("📊 메이플스토리 유저 행동 패턴 분석 대시보드")
    st.divider()

    # 데이터 기본 가공 (Wide -> Long)
    daily_cols = [c for c in df.columns if c.startswith('Daily_')]
    if not daily_cols:
        st.warning("분석할 일일 데이터가 부족합니다.")
        return

    melted = df.melt(id_vars=['segment', 'name'], value_vars=daily_cols, var_name='Date_Col', value_name='Exp')

    # 🚨 [핵심 보정] 실제 사냥 시점과 로그 시점의 시차(-1일) 보정
    melted['Date'] = pd.to_datetime(melted['Date_Col'].str.replace('Daily_', '')) - pd.Timedelta(days=1)

    tab1, tab2, tab3 = st.tabs(["🎯 쇼케이스 영향 (Pre vs Post)", "📅 주간 패턴 (메요일/선데이)", "☀️ 선데이 이벤트 심층 분석"])

    # ==========================================
    # TAB 1: 쇼케이스 영향 분석
    # ==========================================
    with tab1:
        st.markdown("쇼케이스 일자를 기준으로 **동일한 기간(전/후)** 동안의 유저 레벨링 동기 변화를 분석합니다.")

        min_date = melted['Date'].min()
        showcase_dt = pd.to_datetime(SHOWCASE_DATE)
        days_diff = (showcase_dt - min_date).days
        max_date_sym = showcase_dt + pd.Timedelta(days=days_diff)

        df_sym = melted[(melted['Date'] >= min_date) & (melted['Date'] <= max_date_sym)]
        st.info(
            f"**분석 기간:** {min_date.strftime('%Y-%m-%d')} ~ {max_date_sym.strftime('%Y-%m-%d')} (쇼케이스 기준 ±{days_diff}일)")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("전체 유저 평균 성장 추이")
            trend_total = df_sym.groupby('Date')['Exp'].mean().reset_index()
            fig_total = px.line(trend_total, x='Date', y='Exp', markers=True, labels={'Exp': '평균 경험치', 'Date': '날짜'})
            sc_str = showcase_dt.strftime('%Y-%m-%d')
            fig_total.add_vline(x=sc_str, line_width=2, line_dash="dash", line_color="red")
            fig_total.add_annotation(x=sc_str, y=1, yref="paper", text="Showcase", showarrow=False,
                                     font=dict(color="red", size=14))
            st.plotly_chart(fig_total, use_container_width=True)

        with col2:
            st.subheader("레벨 구간별 성장 추이 (평상시 대비 변화율)")
            trend_seg = df_sym.groupby(['segment', 'Date'])['Exp'].mean().reset_index()
            baseline = df.groupby('segment')['Pre_Avg'].mean().reset_index()
            baseline.rename(columns={'Pre_Avg': 'Baseline_Exp'}, inplace=True)
            trend_seg = pd.merge(trend_seg, baseline, on='segment')
            trend_seg['Exp_Ratio'] = (trend_seg['Exp'] / trend_seg['Baseline_Exp']) * 100

            fig_seg = px.line(trend_seg, x='Date', y='Exp_Ratio', color='segment', markers=True,
                              labels={'Exp_Ratio': '평상시 대비 (%)', 'Date': '날짜'})
            fig_seg.add_hline(y=100, line_dash="dot", line_color="gray")
            fig_seg.add_vline(x=sc_str, line_width=2, line_dash="dash", line_color="red")
            fig_seg.add_annotation(x=sc_str, y=1, yref="paper", text="Showcase", showarrow=False,
                                   font=dict(color="red", size=14))
            st.plotly_chart(fig_seg, use_container_width=True)

        st.divider()
        st.subheader("🧪 통계 검증: 쇼케이스 전/후 유의미한 변화가 있었는가?")
        c_bar, c_stat = st.columns([2, 1.5])
        with c_bar:
            summary = df.groupby('segment')[['Pre_Avg', 'Post_Avg']].mean().reset_index()
            summary['Growth_Rate'] = (summary['Post_Avg'] - summary['Pre_Avg']) / summary['Pre_Avg'] * 100
            fig_bar = px.bar(summary, x='segment', y='Growth_Rate', color='segment', title="구간별 성장 증감률 (%)",
                             text_auto='.1f')
            fig_bar.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_stat:
            st.markdown("#### 대응표본 t-검정 결과")
            for seg in sorted(df['segment'].dropna().unique()):
                seg_data = df[df['segment'] == seg].dropna(subset=['Pre_Avg', 'Post_Avg'])
                if len(seg_data) > 1:
                    t_stat, p_val = stats.ttest_rel(seg_data['Pre_Avg'], seg_data['Post_Avg'])
                    is_inc = seg_data['Post_Avg'].mean() > seg_data['Pre_Avg'].mean()
                    dir_icon = "📈 증가" if is_inc else "📉 감소"
                    sig = "매우 유의미함 (⭐⭐⭐)" if p_val < 0.001 else ("유의미함 (⭐)" if p_val < 0.05 else "차이 없음 (❌)")
                    st.info(f"**[{seg}]**\n* 변화: {dir_icon} ({sig})\n* P-value: {p_val:.4e}")

    # ==========================================
    # TAB 2: 메요일 / 선데이 주간 단일 분석
    # ==========================================
    with tab2:
        st.markdown("수요일 시작 7일 단위로 특정 주차와 그룹의 요일별 패턴을 상세 분석합니다.")
        day_order = ['수', '목', '금', '토', '일', '월', '화']
        day_map = {2: '수', 3: '목', 4: '금', 5: '토', 6: '일', 0: '월', 1: '화'}

        melted['DayOfWeek'] = melted['Date'].dt.dayofweek
        shifted_date = melted['Date'] - pd.Timedelta(days=2)
        min_shifted = shifted_date.min()
        melted['Week_Idx'] = ((shifted_date - min_shifted).dt.days // 7) + 1
        melted['Week_Label'] = melted['Week_Idx'].astype(str) + "주차"
        melted['Day_Name'] = melted['DayOfWeek'].map(day_map)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            opts_w = ["전체 주차 (평균)"] + sorted(list(melted['Week_Label'].unique()),
                                             key=lambda x: int(x.replace("주차", "")))
            sel_w = st.selectbox("📅 주차 선택:", opts_w)
        with col_s2:
            opts_s = ["전체 유저"] + list(melted['segment'].dropna().unique())
            sel_s = st.selectbox("📊 그룹 선택:", opts_s)

        f_df = melted.copy()
        if sel_w != "전체 주차 (평균)": f_df = f_df[f_df['Week_Label'] == sel_w]
        if sel_s != "전체 유저": f_df = f_df[f_df['segment'] == sel_s]

        t_weekly = f_df.groupby('Day_Name')['Exp'].mean().reset_index()

        # 🚨 [중요] 거미줄 현상 방지: 데이터프레임 자체를 요일 순서대로 완벽하게 정렬
        t_weekly['Day_Name'] = pd.Categorical(t_weekly['Day_Name'], categories=day_order, ordered=True)
        t_weekly = t_weekly.sort_values('Day_Name')

        fig_week = px.line(t_weekly, x='Day_Name', y='Exp', markers=True, title=f"[{sel_w}] {sel_s} 요일별 패턴")

        fig_week.add_vrect(x0=-0.2, x1=2.2, fillcolor="LightSteelBlue", opacity=0.3, layer="below", line_width=0)
        y_m = t_weekly['Exp'].max() if not t_weekly.empty else 1
        fig_week.add_vline(x=1, line_width=1.5, line_dash="dot", line_color="orange")
        fig_week.add_annotation(x=1, y=y_m * 0.1, text="목(메요일)", showarrow=False, font=dict(color="orange"),
                                textangle=-90)
        fig_week.add_vline(x=4, line_width=1.5, line_dash="dot", line_color="green")
        fig_week.add_annotation(x=4, y=y_m * 0.1, text="일(선데이)", showarrow=False, font=dict(color="green"),
                                textangle=-90)
        st.plotly_chart(fig_week, use_container_width=True)

    # ==========================================
    # TAB 3: 선데이 메이플 이벤트 심층 분석
    # ==========================================
    with tab3:
        st.markdown("선데이 메이플의 혜택 성격(경타포스/사냥/강화/기타)에 따라 유저들의 실제 사냥량이 어떻게 달라지는지 검증합니다.")

        sunday_log_df = load_sunday_log()
        df_sunday_only = melted[melted['DayOfWeek'] == 6].copy()

        # 🚨 [누락 방어] 실제 수집된 데이터(df_sunday_only)를 기준으로 Left Join
        df_sunday_merged = pd.merge(df_sunday_only, sunday_log_df, on='Date', how='left')

        # 대분류 적용 (4가지 그룹)
        df_sunday_merged['Event_Category'] = df_sunday_merged['Sunday_Type'].apply(classify_sunday_event)

        # 대시보드 표시용 요약 테이블 생성
        display_log = df_sunday_merged[['Date', 'Sunday_Type', 'Event_Category']].drop_duplicates().sort_values('Date')
        display_log['Date'] = display_log['Date'].dt.strftime('%Y-%m-%d')
        display_log['Sunday_Type'] = display_log['Sunday_Type'].fillna('기록없음(일반)')

        col_sub1, col_sub2 = st.columns([1, 2.5])
        with col_sub1:
            st.info("📌 **실제 수집된 선데이 분류 현황**")
            st.dataframe(display_log.reset_index(drop=True), use_container_width=True)
            sel_seg_sun = st.selectbox("📊 분석 그룹 선택:", ["전체 유저"] + list(df_sunday_merged['segment'].dropna().unique()),
                                       key='sun_seg')

        with col_sub2:
            # 선택한 그룹으로 필터링
            target_df = df_sunday_merged if sel_seg_sun == "전체 유저" else df_sunday_merged[
                df_sunday_merged['segment'] == sel_seg_sun]

            # 카테고리별 정렬 순서 고정
            cat_order = ['경타포스', '사냥', '강화', '기타']

            # 🚨 [100% 원본 데이터 사용] 샘플링 없이 유저 전체 데이터를 사용하여 박스플롯 그림
            fig_box = px.box(target_df, x='Event_Category', y='Exp', color='Event_Category',
                             title=f"[{sel_seg_sun}] 선데이 이벤트 유형별 획득 경험치 분포",
                             category_orders={'Event_Category': cat_order},
                             labels={'Event_Category': '이벤트 분류', 'Exp': '획득 경험치'})
            st.plotly_chart(fig_box, use_container_width=True)

        st.divider()
        st.subheader("🧪 통계 검증: 보상 퀄리티에 따른 사냥량 차이 (ANOVA 검정)")

        # ANOVA 통계 검정 준비
        groups = {}
        for cat in cat_order:
            data_series = target_df[target_df['Event_Category'] == cat]['Exp'].dropna()
            if not data_series.empty:
                groups[cat] = data_series

        if len(groups) >= 2:
            # 그룹이 2개 이상일 때만 분산분석(ANOVA) 실행
            f_stat, p_val = stats.f_oneway(*groups.values())

            # 평균값 계산 및 표시
            means_text = " | ".join([f"**{k}**: {v.mean():.1e}" for k, v in groups.items()])

            if p_val < 0.05:
                st.success(f"**검증 결과: 집단 간 유의미한 사냥량 차이가 존재합니다! (P-value: {p_val:.4e})**\n\n"
                           f"선데이 이벤트의 종류(보상)에 따라 유저들의 사냥 동기가 뚜렷하게 달라졌음을 시사합니다.\n\n"
                           f"*(평균 경험치 획득량 👉 {means_text})*")
            else:
                st.info(f"**검증 결과: 집단 간 통계적으로 유의미한 차이가 없습니다. (P-value: {p_val:.4f})**\n\n"
                        f"이벤트의 종류보다는 요일 자체(일요일)의 특성이 사냥량에 더 큰 영향을 미쳤거나, 표본 크기가 부족할 수 있습니다.")
        else:
            st.warning("분석할 수 있는 선데이 이벤트 그룹이 2개 미만입니다. 더 많은 날짜의 데이터가 필요합니다.")


if __name__ == "__main__":
    main()