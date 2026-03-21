import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ================= CONFIG =================
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_PATH  = os.path.join(BASE_DIR, "data", "showcase", "preprocessed", "daily_segment_processed.csv")
SUNDAY_LOG_PATH = os.path.join(BASE_DIR, "data", "showcase", "sundaylog.txt")
SHOWCASE_DATE   = "2025-12-13"
MIN_VALID_DAYS  = 7
# ==========================================

pd.set_option('display.float_format', '{:,.0f}'.format)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

SEP  = "=" * 70
SEP2 = "-" * 70


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subsection(title):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)


# ── 데이터 로드 ────────────────────────────────────────────────────────────
section("데이터 로드")

if not os.path.exists(PROCESSED_PATH):
    print(f"[오류] 전처리 파일 없음: {PROCESSED_PATH}")
    print("       B_preprocessing.py 를 먼저 실행해주세요.")
    exit()

df = pd.read_csv(PROCESSED_PATH)
print(f"총 유저 수       : {len(df):,}명")
print(f"레벨 구간 분포   :\n{df['segment'].value_counts().sort_index().to_string()}")

daily_cols = sorted(
    [c for c in df.columns if c.startswith('Daily_')],
    key=lambda x: pd.to_datetime(x.replace('Daily_', ''))
)

# Wide → Long
melted = df.melt(
    id_vars=['segment', 'name', 'job', 'world'],
    value_vars=daily_cols,
    var_name='Date_Col',
    value_name='Exp'
)
melted['Date']      = pd.to_datetime(melted['Date_Col'].str.replace('Daily_', '')) - pd.Timedelta(days=1)
melted['DayOfWeek'] = melted['Date'].dt.dayofweek

shifted_date       = melted['Date'] - pd.Timedelta(days=2)
melted['Week_Idx'] = ((shifted_date - shifted_date.min()).dt.days // 7) + 1

showcase_dt = pd.to_datetime(SHOWCASE_DATE)
min_date    = melted['Date'].min()
max_date    = melted['Date'].max()
pre_days    = (showcase_dt - min_date).days
post_days   = (max_date - showcase_dt).days
sym_days    = min(pre_days, post_days)
sym_start   = showcase_dt - pd.Timedelta(days=sym_days)
sym_end     = showcase_dt + pd.Timedelta(days=sym_days)

print(f"\n전체 수집 기간   : {min_date.date()} ~ {max_date.date()}")
print(f"쇼케이스 날짜    : {SHOWCASE_DATE}")
print(f"대칭 분석 기간   : {sym_start.date()} ~ {sym_end.date()}  (±{sym_days}일)")


# ══════════════════════════════════════════════════════════════════════
# [1] 쇼케이스 영향 분석
# ══════════════════════════════════════════════════════════════════════
section("[1] 쇼케이스 영향 분석 - Pre vs Post (대칭 ±{}일)".format(sym_days))

# 1-1. 날짜별 전체 평균 성장 추이 (대칭 기간)
subsection("1-1. 전체 유저 일별 평균 경험치 추이")

df_sym     = melted[(melted['Date'] >= sym_start) & (melted['Date'] <= sym_end)]
trend_total = df_sym.groupby('Date')['Exp'].mean().reset_index()
trend_total['구분'] = trend_total['Date'].apply(lambda d: 'POST' if d > showcase_dt else 'PRE ')
trend_total['Date'] = trend_total['Date'].dt.strftime('%Y-%m-%d')
trend_total.columns = ['날짜', '평균경험치', '구분']
print(trend_total.to_string(index=False))

# 1-2. 구간별 Pre/Post 평균
subsection("1-2. 레벨 구간별 Pre/Post 평균 및 변화율")

sym_pre_cols  = [c for c in daily_cols
                 if sym_start < pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
sym_post_cols = [c for c in daily_cols
                 if showcase_dt < pd.to_datetime(c.replace('Daily_', '')) <= sym_end]

df_stat = df.copy()
df_stat['Sym_Pre_Avg']    = df_stat[sym_pre_cols].mean(axis=1)        if sym_pre_cols  else np.nan
df_stat['Sym_Post_Avg']   = df_stat[sym_post_cols].mean(axis=1)       if sym_post_cols else np.nan
df_stat['Sym_Pre_Valid']  = df_stat[sym_pre_cols].notna().sum(axis=1) if sym_pre_cols  else 0
df_stat['Sym_Post_Valid'] = df_stat[sym_post_cols].notna().sum(axis=1) if sym_post_cols else 0

df_stat = df_stat[
    (df_stat['Sym_Pre_Valid']  >= MIN_VALID_DAYS) &
    (df_stat['Sym_Post_Valid'] >= MIN_VALID_DAYS)
]
print(f"(유효 데이터 {MIN_VALID_DAYS}일 미만 제외 후 {len(df_stat):,}명 사용)\n")

summary = df_stat.groupby('segment')[['Sym_Pre_Avg', 'Sym_Post_Avg']].mean()
summary['변화율(%)'] = (summary['Sym_Post_Avg'] - summary['Sym_Pre_Avg']) / summary['Sym_Pre_Avg'] * 100
summary['n']         = df_stat.groupby('segment').size()
summary.columns      = ['Pre 평균', 'Post 평균', '변화율(%)', 'n']
print(summary.to_string())

# 1-3. 대응표본 t-검정 (구간별)
subsection("1-3. 대응표본 t-검정 (구간별)")
print(f"{'구간':<14} {'n':>5}  {'Pre 평균':>14}  {'Post 평균':>14}  {'변화율(%)':>9}  {'t-stat':>8}  {'p-value':>12}  판정")
print("-" * 90)

for seg in sorted(df_stat['segment'].dropna().unique()):
    sd = df_stat[df_stat['segment'] == seg].dropna(subset=['Sym_Pre_Avg', 'Sym_Post_Avg'])
    if len(sd) < 2:
        continue
    t_stat_val, p_val = stats.ttest_rel(sd['Sym_Pre_Avg'], sd['Sym_Post_Avg'])
    pre_m  = sd['Sym_Pre_Avg'].mean()
    post_m = sd['Sym_Post_Avg'].mean()
    rate   = (post_m - pre_m) / pre_m * 100
    sig    = "★★★ p<0.001" if p_val < 0.001 else ("★ p<0.05" if p_val < 0.05 else "비유의")
    arrow  = "▲" if post_m > pre_m else "▼"
    print(f"{seg:<14} {len(sd):>5}  {pre_m:>14,.0f}  {post_m:>14,.0f}  {rate:>+8.1f}%  "
          f"{t_stat_val:>8.3f}  {p_val:>12.4e}  {arrow} {sig}")


# ══════════════════════════════════════════════════════════════════════
# [2] 주간 패턴 (요일별 사냥 분포)
# ══════════════════════════════════════════════════════════════════════
section("[2] 주간 패턴 분석")

day_map   = {2: '수', 3: '목', 4: '금', 5: '토', 6: '일', 0: '월', 1: '화'}
day_order = ['수', '목', '금', '토', '일', '월', '화']
melted['요일'] = melted['DayOfWeek'].map(day_map)

# 2-1. 전체 평균 요일별 경험치
subsection("2-1. 전체 유저 요일별 평균 경험치")

day_avg = melted.groupby('요일')['Exp'].mean().reindex(day_order)
day_avg.name = '평균 경험치'
print(day_avg.to_string())
print(f"\n  → 최고 요일: {day_avg.idxmax()}  ({day_avg.max():,.0f})")
print(f"  → 최저 요일: {day_avg.idxmin()}  ({day_avg.min():,.0f})")
print(f"  → 목(메요일)/일(선데이) 비율: "
      f"목={day_avg['목']/day_avg.mean()*100:.1f}%  일={day_avg['일']/day_avg.mean()*100:.1f}% (전체 평균 대비)")

# 2-2. 구간별 요일 패턴
subsection("2-2. 레벨 구간별 요일별 평균 경험치")

day_seg = (
    melted.groupby(['segment', '요일'])['Exp']
    .mean()
    .unstack('요일')
    .reindex(columns=day_order)
    .reindex(sorted(melted['segment'].dropna().unique()))
)
print(day_seg.to_string())

# 2-3. 주차 × 요일 피벗 (전체)
subsection("2-3. 주차 × 요일 평균 경험치 피벗 (전체 유저)")

pivot = melted.pivot_table(index='Week_Idx', columns='요일', values='Exp', aggfunc='mean')
pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
pivot.index.name = '주차'
showcase_week = int(((showcase_dt - pd.Timedelta(days=2) - (melted['Date'] - pd.Timedelta(days=2)).min()).days) // 7) + 1
print(pivot.to_string())
print(f"\n  → 쇼케이스 주차: {showcase_week}주차")


# ══════════════════════════════════════════════════════════════════════
# [3] 선데이 메이플 이벤트 분석
# ══════════════════════════════════════════════════════════════════════
section("[3] 선데이 메이플 이벤트 분석 - F-검정 (ANOVA)")

# 선데이 로그 로드
sunday_log_df = pd.DataFrame(columns=['Date', 'Sunday_Type'])
if os.path.exists(SUNDAY_LOG_PATH):
    rows = []
    with open(SUNDAY_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                date_str, event_type = line.strip().split(':', 1)
                rows.append({'Date': pd.to_datetime(date_str.strip()), 'Sunday_Type': event_type.strip()})
    sunday_log_df = pd.DataFrame(rows)

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

# 누락 선데이 체크
all_sundays = sorted(melted[melted['DayOfWeek'] == 6]['Date'].dt.normalize().unique())
logged_dates = set(sunday_log_df['Date'].dt.normalize()) if not sunday_log_df.empty else set()
missing = [d for d in all_sundays if d not in logged_dates]
if missing:
    print(f"[주의] sundaylog 미등록 선데이 {len(missing)}개: "
          + ", ".join(pd.Timestamp(d).strftime('%Y-%m-%d') for d in missing))

# 일요일 데이터 + 이벤트 병합
df_sun = melted[melted['DayOfWeek'] == 6].copy()
df_sun = pd.merge(df_sun, sunday_log_df, on='Date', how='left')
df_sun['Event_Category'] = df_sun['Sunday_Type'].apply(classify_sunday_event)

# 유저별 Pre_Avg 대비 변화율
pre_avg_map = df.set_index('name')['Pre_Avg']
df_sun['Pre_Avg'] = df_sun['name'].map(pre_avg_map)
valid = df_sun['Pre_Avg'].notna() & (df_sun['Pre_Avg'] > 0) & df_sun['Exp'].notna()
df_sun['Exp_Ratio'] = np.where(valid, (df_sun['Exp'] / df_sun['Pre_Avg']) * 100, np.nan)

# 3-1. 선데이 분류 현황
subsection("3-1. 선데이 분류 현황")
log_display = (
    df_sun[['Date', 'Sunday_Type', 'Event_Category']]
    .drop_duplicates()
    .sort_values('Date')
)
log_display['Date'] = log_display['Date'].dt.strftime('%Y-%m-%d')
log_display['Sunday_Type'] = log_display['Sunday_Type'].fillna('기록없음(일반)')
print(log_display.to_string(index=False))

# 3-2. 이벤트 유형별 변화율 통계
subsection("3-2. 이벤트 유형별 평상시 대비 변화율 통계")

cat_order = ['경타포스', '사냥', '강화', '기타']
cat_stats = (
    df_sun.groupby('Event_Category')['Exp_Ratio']
    .agg(n='count', mean='mean', median='median', std='std',
         q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
    .reindex(cat_order)
    .dropna(how='all')
)
cat_stats.columns = ['n', '평균(%)', '중앙값(%)', '표준편차', 'Q25', 'Q75']
print(cat_stats.to_string(float_format='{:.2f}'.format))
print("\n  (100% = 평상시와 동일, >100% = 평상시보다 많이 사냥)")

# 3-3. ANOVA F-검정
subsection("3-3. ANOVA F-검정")
print("귀무가설: 모든 이벤트 유형에서 평상시 대비 사냥량 변화율이 동일하다.\n")

q99 = df_sun['Exp_Ratio'].quantile(0.99)
groups = {}
for cat in cat_order:
    s = df_sun[df_sun['Event_Category'] == cat]['Exp_Ratio'].dropna()
    s = s[s <= q99]
    if len(s) > 1:
        groups[cat] = s

if len(groups) >= 2:
    f_stat_val, p_val = stats.f_oneway(*groups.values())
    print(f"F-statistic : {f_stat_val:.4f}")
    print(f"P-value     : {p_val:.4e}")
    print(f"결론        : {'유의미한 차이 있음 ✅ (p<0.05)' if p_val < 0.05 else '유의미한 차이 없음 ❌'}")
    print(f"\n그룹별 평균 변화율:")
    for k, v in groups.items():
        print(f"  {k:<10} : {v.mean():>6.1f}%  (n={len(v):,})")

    # 3-4. Tukey HSD 사후 검정
    if len(groups) >= 3:
        subsection("3-4. Tukey HSD 사후 검정 (쌍별 비교)")
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            all_vals   = np.concatenate(list(groups.values()))
            all_labels = np.concatenate([[k] * len(v) for k, v in groups.items()])
            tukey      = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
            tukey_df   = pd.DataFrame(
                data=tukey._results_table.data[1:],
                columns=tukey._results_table.data[0]
            )
            tukey_df['유의미'] = tukey_df['reject'].apply(lambda x: "✅ 차이 있음" if x else "❌ 차이 없음")
            print(tukey_df[['group1', 'group2', 'meandiff', 'p-adj', '유의미']].to_string(index=False))
        except ImportError:
            print("[주의] statsmodels 미설치 → pip install statsmodels")
else:
    print("[주의] 분석 가능한 그룹이 2개 미만입니다.")

# 3-5. 구간별 × 이벤트 유형 교차 분석
subsection("3-5. 레벨 구간 × 이벤트 유형 평균 변화율 교차표")
cross = (
    df_sun[df_sun['Event_Category'].isin(cat_order)]
    .groupby(['segment', 'Event_Category'])['Exp_Ratio']
    .mean()
    .unstack('Event_Category')
    .reindex(columns=[c for c in cat_order if c in df_sun['Event_Category'].unique()])
)
print(cross.to_string(float_format='{:.1f}'.format))


# ══════════════════════════════════════════════════════════════════════
# [4] 직업군 반응 비교 (상위/하위 10개)
# ══════════════════════════════════════════════════════════════════════
section("[4] 직업군별 쇼케이스 반응 비교")

df_job = df[
    (df['Pre_Valid_Days']  >= MIN_VALID_DAYS) &
    (df['Post_Valid_Days'] >= MIN_VALID_DAYS)
].copy()
df_job = df_job.dropna(subset=['Pre_Avg', 'Post_Avg'])

job_summary = df_job.groupby('job').agg(
    n=('name', 'count'),
    pre_avg=('Pre_Avg', 'mean'),
    post_avg=('Post_Avg', 'mean')
).reset_index()
job_summary['변화율(%)'] = (job_summary['post_avg'] - job_summary['pre_avg']) / job_summary['pre_avg'] * 100
job_summary = job_summary[job_summary['n'] >= 5].sort_values('변화율(%)', ascending=False)

subsection("4-1. 변화율 상위 10개 직업군 (n≥5)")
print(job_summary.head(10)[['job', 'n', 'pre_avg', 'post_avg', '변화율(%)']].to_string(index=False))

subsection("4-2. 변화율 하위 10개 직업군 (n≥5)")
print(job_summary.tail(10)[['job', 'n', 'pre_avg', 'post_avg', '변화율(%)']].to_string(index=False))

print(f"\n{SEP}")
print("  분석 완료")
print(SEP)
