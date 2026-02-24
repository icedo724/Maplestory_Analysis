import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
import os

# ================= CONFIG =================
st.set_page_config(page_title="메이플 레벨별 성장 분석", page_icon="📊", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))

# 전처리된 데이터 경로 (preprocessed 폴더)
PROCESSED_PATH = os.path.join(base_dir, "data", "showcase", "preprocessed", "daily_segment_processed.csv")
SHOWCASE_DATE = "2025-12-13"


@st.cache_data
def load_data():
    if not os.path.exists(PROCESSED_PATH): return None
    df = pd.read_csv(PROCESSED_PATH)
    return df


def main():
    df = load_data()
    if df is None:
        st.error(f"데이터 파일이 없습니다. 전처리 코드를 먼저 실행해주세요: {PROCESSED_PATH}")
        return

    st.title("📊 메이플스토리 유저 행동 패턴 분석 대시보드")
    st.divider()

    # 데이터 기본 가공 (Wide -> Long)
    daily_cols = [c for c in df.columns if c.startswith('Daily_')]
    if not daily_cols:
        st.warning("분석할 일일 데이터가 부족합니다.")
        return

    melted = df.melt(id_vars=['segment', 'name'], value_vars=daily_cols, var_name='Date_Col', value_name='Exp')
    melted['Date'] = pd.to_datetime(melted['Date_Col'].str.replace('Daily_', ''))

    # 탭 생성
    tab1, tab2 = st.tabs(["🎯 쇼케이스 영향 분석 (Pre vs Post)", "📅 주간 요일별 패턴 분석 (메요일/선데이)"])

    # ==========================================
    # TAB 1: 쇼케이스 영향 분석
    # ==========================================
    with tab1:
        st.markdown("쇼케이스 일자를 기준으로 **동일한 기간(전/후)** 동안의 유저 레벨링 동기 변화를 분석합니다.")

        # 쇼케이스 전후 대칭 기간 설정
        min_date = melted['Date'].min()
        showcase_dt = pd.to_datetime(SHOWCASE_DATE)
        days_diff = (showcase_dt - min_date).days
        max_date_sym = showcase_dt + pd.Timedelta(days=days_diff)

        df_sym = melted[(melted['Date'] >= min_date) & (melted['Date'] <= max_date_sym)]
        st.info(
            f"**분석 기간:** {min_date.strftime('%Y-%m-%d')} ~ {max_date_sym.strftime('%Y-%m-%d')} (쇼케이스 기준 ±{days_diff}일)")

        col1, col2 = st.columns(2)

        # 1. 전체 유저 평균 성장 추이 (절대 수치)
        with col1:
            st.subheader("전체 유저 평균 성장 추이 (경험치)")
            trend_total = df_sym.groupby('Date')['Exp'].mean().reset_index()
            fig_total = px.line(trend_total, x='Date', y='Exp', markers=True,
                                labels={'Exp': '평균 경험치 획득량', 'Date': '날짜'})

            # 쇼케이스 선 표기 (문자열 변환으로 에러 방지)
            sc_str = showcase_dt.strftime('%Y-%m-%d')
            fig_total.add_vline(x=sc_str, line_width=2, line_dash="dash", line_color="red")
            fig_total.add_annotation(x=sc_str, y=1, yref="paper", text="Showcase", showarrow=False,
                                     font=dict(color="red", size=14))
            st.plotly_chart(fig_total, use_container_width=True)

        # 2. 레벨 구간별 성장 추이 (변화율 - 미분 개념 적용)
        with col2:
            st.subheader("레벨 구간별 성장 추이 (평상시 대비 변화율)")
            trend_seg = df_sym.groupby(['segment', 'Date'])['Exp'].mean().reset_index()

            # 구간별 기준점(쇼케이스 전 평균) 결합
            baseline = df.groupby('segment')['Pre_Avg'].mean().reset_index()
            baseline.rename(columns={'Pre_Avg': 'Baseline_Exp'}, inplace=True)
            trend_seg = pd.merge(trend_seg, baseline, on='segment')

            # 평상시(100%) 대비 획득 비율 계산
            trend_seg['Exp_Ratio'] = (trend_seg['Exp'] / trend_seg['Baseline_Exp']) * 100

            fig_seg = px.line(trend_seg, x='Date', y='Exp_Ratio', color='segment', markers=True,
                              labels={'Exp_Ratio': '평상시 대비 획득량 (%)', 'Date': '날짜', 'segment': '레벨 구간'})

            # 100% 기준선 및 쇼케이스 선
            fig_seg.add_hline(y=100, line_dash="dot", line_color="gray")
            fig_seg.add_vline(x=sc_str, line_width=2, line_dash="dash", line_color="red")
            fig_seg.add_annotation(x=sc_str, y=1, yref="paper", text="Showcase", showarrow=False,
                                   font=dict(color="red", size=14))
            st.plotly_chart(fig_seg, use_container_width=True)

        st.divider()
        st.subheader("🧪 통계 검증: 쇼케이스 전/후 유의미한 변화가 있었는가? (Paired T-test)")

        c_bar, c_stat = st.columns([2, 1.5])

        with c_bar:
            summary = df.groupby('segment')[['Pre_Avg', 'Post_Avg']].mean()
            summary['Growth_Rate'] = (summary['Post_Avg'] - summary['Pre_Avg']) / summary['Pre_Avg'] * 100
            summary = summary.reset_index()
            fig_bar = px.bar(summary, x='segment', y='Growth_Rate', color='segment',
                             title="구간별 성장 속도 증감률 (%)", text_auto='.1f')
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
                else:
                    st.warning(f"[{seg}] 데이터 부족")

    # ==========================================
    # TAB 2: 메요일 / 선데이 주간 단일 분석
    # ==========================================
    with tab2:
        st.markdown("수요일 시작 7일 단위로 특정 주차와 그룹의 요일별 패턴을 상세 분석합니다.")

        # 주차 및 요일 계산
        day_order = ['수', '목', '금', '토', '일', '월', '화']
        day_map = {2: '수', 3: '목', 4: '금', 5: '토', 6: '일', 0: '월', 1: '화'}

        melted['DayOfWeek'] = melted['Date'].dt.dayofweek
        shifted_date = melted['Date'] - pd.Timedelta(days=2)
        min_shifted = shifted_date.min()
        melted['Week_Idx'] = ((shifted_date - min_shifted).dt.days // 7) + 1
        melted['Week_Label'] = melted['Week_Idx'].astype(str) + "주차"
        melted['Day_Name'] = melted['DayOfWeek'].map(day_map)

        # 콤보박스 필터
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            opts_w = ["전체 주차 (평균)"] + sorted(list(melted['Week_Label'].unique()),
                                             key=lambda x: int(x.replace("주차", "")))
            sel_w = st.selectbox("📅 분석할 주차 선택:", opts_w)
        with col_s2:
            opts_s = ["전체 유저"] + list(melted['segment'].dropna().unique())
            sel_s = st.selectbox("📊 분석할 그룹 선택:", opts_s)

        # 필터링 및 요일 정렬
        f_df = melted.copy()
        if sel_w != "전체 주차 (평균)": f_df = f_df[f_df['Week_Label'] == sel_w]
        if sel_s != "전체 유저": f_df = f_df[f_df['segment'] == sel_s]

        # 데이터 정렬 (꼬임 방지 핵심)
        t_weekly = f_df.groupby('Day_Name')['Exp'].mean().reset_index()
        t_weekly['Day_Name'] = pd.Categorical(t_weekly['Day_Name'], categories=day_order, ordered=True)
        t_weekly = t_weekly.sort_values('Day_Name')

        fig_week = px.line(t_weekly, x='Day_Name', y='Exp', markers=True,
                           title=f"[{sel_w}] {sel_s} 요일별 패턴", labels={'Exp': '평균 경험치'})

        # 수~금 하이라이트 및 요일선
        fig_week.add_vrect(x0=-0.2, x1=2.2, fillcolor="LightSteelBlue", opacity=0.3, layer="below", line_width=0)
        y_m = t_weekly['Exp'].max() if not t_weekly.empty else 1

        fig_week.add_vline(x=1, line_width=1.5, line_dash="dot", line_color="orange")
        fig_week.add_annotation(x=1, y=y_m * 0.1, text="목(메요일)", showarrow=False, font=dict(color="orange"),
                                textangle=-90)
        fig_week.add_vline(x=4, line_width=1.5, line_dash="dot", line_color="green")
        fig_week.add_annotation(x=4, y=y_m * 0.1, text="일(선데이)", showarrow=False, font=dict(color="green"),
                                textangle=-90)

        st.plotly_chart(fig_week, use_container_width=True)


if __name__ == "__main__":
    main()