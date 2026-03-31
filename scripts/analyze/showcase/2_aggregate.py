"""
집계 데이터 내보내기 스크립트
원본 전처리 데이터(152MB)를 Streamlit Cloud 호스팅용 집계 파일(~50KB)로 변환.
생성 파일 → data/processed/showcase/aggregated/
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ================= CONFIG =================
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_PATH  = os.path.join(BASE_DIR, "data", "processed", "showcase", "daily_segment_processed.csv")
SUNDAY_LOG_PATH = os.path.join(BASE_DIR, "data", "meta", "sundaylog.txt")
AGG_DIR         = os.path.join(BASE_DIR, "data", "processed", "showcase", "aggregated")
SHOWCASE_DATE   = "2025-12-13"
EVENT_LOG_PATH  = os.path.join(BASE_DIR, "data", "meta", "eventlog.txt")
MIN_VALID_DAYS  = 7
# ==========================================

if not os.path.exists(PROCESSED_PATH):
    print("[오류] 전처리 파일 없음. 1_preprocess.py 먼저 실행하세요.")
    sys.exit()

os.makedirs(AGG_DIR, exist_ok=True)

print("[로드] 전처리 데이터 로드 중...")
df = pd.read_csv(PROCESSED_PATH)
daily_cols = sorted([c for c in df.columns if c.startswith('Daily_')],
                    key=lambda x: pd.to_datetime(x.replace('Daily_', '')))

melted = df.melt(id_vars=['segment', 'name', 'job', 'world'], value_vars=daily_cols,
                 var_name='Date_Col', value_name='Exp')
melted['Date']      = pd.to_datetime(melted['Date_Col'].str.replace('Daily_', '')) - pd.Timedelta(days=1)
melted['DayOfWeek'] = melted['Date'].dt.dayofweek
shifted             = melted['Date'] - pd.Timedelta(days=2)
melted['Week_Idx']  = ((shifted - shifted.min()).dt.days // 7) + 1

showcase_dt = pd.to_datetime(SHOWCASE_DATE)
min_date    = melted['Date'].min()
max_date    = melted['Date'].max()
sym_days    = min((showcase_dt - min_date).days, (max_date - showcase_dt).days)
sym_start   = showcase_dt - pd.Timedelta(days=sym_days)
sym_end     = showcase_dt + pd.Timedelta(days=sym_days)


# ── 1. 날짜 × 구간별 일평균 경험치 ────────────────────────────────────────
print("[1/8] agg_daily_segment 생성 중...")
agg1 = melted.groupby(['Date', 'segment'])['Exp'].mean().reset_index()
agg1['Date'] = agg1['Date'].dt.strftime('%Y-%m-%d')
agg1.columns = ['date', 'segment', 'avg_exp']
agg1.to_csv(os.path.join(AGG_DIR, 'agg_daily_segment.csv'), index=False, encoding='utf-8-sig')
print(f"     → {len(agg1)}행 저장")


# ── 2. 구간별 Pre/Post 요약 + t-검정 결과 ─────────────────────────────────
print("[2/8] agg_segment_summary 생성 중...")
sym_pre_cols  = [c for c in daily_cols if sym_start < pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
sym_post_cols = [c for c in daily_cols if showcase_dt < pd.to_datetime(c.replace('Daily_', '')) <= sym_end]

df_stat = df.copy()
df_stat['Sym_Pre_Avg']    = df_stat[sym_pre_cols].mean(axis=1)
df_stat['Sym_Post_Avg']   = df_stat[sym_post_cols].mean(axis=1)
df_stat['Sym_Pre_Valid']  = df_stat[sym_pre_cols].notna().sum(axis=1)
df_stat['Sym_Post_Valid'] = df_stat[sym_post_cols].notna().sum(axis=1)
df_stat = df_stat[(df_stat['Sym_Pre_Valid'] >= MIN_VALID_DAYS) & (df_stat['Sym_Post_Valid'] >= MIN_VALID_DAYS)]

rows = []
for seg in sorted(df_stat['segment'].dropna().unique()):
    sd = df_stat[df_stat['segment'] == seg].dropna(subset=['Sym_Pre_Avg', 'Sym_Post_Avg'])
    t, p = stats.ttest_rel(sd['Sym_Pre_Avg'], sd['Sym_Post_Avg'])
    rows.append({
        'segment':    seg,
        'n':          len(sd),
        'pre_avg':    sd['Sym_Pre_Avg'].mean(),
        'post_avg':   sd['Sym_Post_Avg'].mean(),
        'growth_rate': (sd['Sym_Post_Avg'].mean() - sd['Sym_Pre_Avg'].mean()) / sd['Sym_Pre_Avg'].mean() * 100,
        't_stat':     t,
        'p_value':    p,
        'sym_days':   sym_days,
        'data_start': min_date.strftime('%Y-%m-%d'),
        'data_end':   max_date.strftime('%Y-%m-%d'),
    })
agg2 = pd.DataFrame(rows)
agg2.to_csv(os.path.join(AGG_DIR, 'agg_segment_summary.csv'), index=False, encoding='utf-8-sig')
print(f"     → {len(agg2)}행 저장")


# ── 3. 썬데이 이벤트별 변화율 집계 ───────────────────────────────────────
print("[3/8] 썬데이 데이터 준비 중...")
sunday_rows = []
if os.path.exists(SUNDAY_LOG_PATH):
    with open(SUNDAY_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                d, e = line.strip().split(':', 1)
                sunday_rows.append({'Date': pd.to_datetime(d.strip()), 'Sunday_Type': e.strip()})
sunday_log = pd.DataFrame(sunday_rows)

def classify_sunday_event(event_str):
    if pd.isna(event_str): return '사냥 외'
    s = str(event_str)
    if '경타포스' in s: return '경타포스'
    if any(k in s for k in ['트레져', '룬콤보', '솔에르다', '사냥']): return '사냥'
    return '사냥 외'

df_sun = melted[melted['DayOfWeek'] == 6].copy()
if not sunday_log.empty:
    df_sun = pd.merge(df_sun, sunday_log, on='Date', how='left')
else:
    df_sun['Sunday_Type'] = np.nan
df_sun['Event_Category'] = df_sun['Sunday_Type'].apply(classify_sunday_event)

pre_avg_map       = df.set_index('name')['Pre_Avg']
df_sun['Pre_Avg'] = df_sun['name'].map(pre_avg_map)
valid             = df_sun['Pre_Avg'].notna() & (df_sun['Pre_Avg'] > 0) & df_sun['Exp'].notna()
df_sun['Exp_Ratio'] = np.where(valid, df_sun['Exp'] / df_sun['Pre_Avg'] * 100, np.nan)
q99 = df_sun['Exp_Ratio'].quantile(0.99)
df_sun_clean = df_sun[df_sun['Exp_Ratio'] <= q99].dropna(subset=['Exp_Ratio'])

# 3a. 날짜 × 구간 × 이벤트 요약
print("     agg_sunday_events 생성 중...")
agg3 = (df_sun_clean
        .groupby(['Date', 'segment', 'Event_Category', 'Sunday_Type'])['Exp_Ratio']
        .agg(mean='mean', median='median', n='count', std='std')
        .reset_index())
agg3['Date'] = agg3['Date'].dt.strftime('%Y-%m-%d')
agg3['Sunday_Type'] = agg3['Sunday_Type'].fillna('기록없음')
agg3.to_csv(os.path.join(AGG_DIR, 'agg_sunday_events.csv'), index=False, encoding='utf-8-sig')
print(f"     → {len(agg3)}행 저장")

# 3b. 박스플롯용 분위수 (go.Box precomputed 방식)
print("[4/8] agg_sunday_box 생성 중...")
cat_order = ['경타포스', '사냥', '사냥 외']
box_rows = []
for scope, subset in [('전체', df_sun_clean)] + [(seg, df_sun_clean[df_sun_clean['segment'] == seg])
                                                   for seg in sorted(df_sun_clean['segment'].dropna().unique())]:
    for cat in cat_order:
        s = subset[subset['Event_Category'] == cat]['Exp_Ratio'].dropna()
        if len(s) < 5:
            continue
        box_rows.append({
            'scope': scope, 'category': cat, 'n': len(s),
            'mean':   s.mean(),
            'p5':     s.quantile(0.05), 'p25':  s.quantile(0.25),
            'median': s.median(),
            'p75':    s.quantile(0.75), 'p95':  s.quantile(0.95),
        })
agg4 = pd.DataFrame(box_rows)
agg4.to_csv(os.path.join(AGG_DIR, 'agg_sunday_box.csv'), index=False, encoding='utf-8-sig')
print(f"     → {len(agg4)}행 저장")

# 3c. ANOVA + Tukey HSD
print("[5/8] agg_anova + agg_tukey 생성 중...")
anova_rows = []
tukey_rows = []

for scope, subset in [('전체', df_sun_clean)] + [(seg, df_sun_clean[df_sun_clean['segment'] == seg])
                                                   for seg in sorted(df_sun_clean['segment'].dropna().unique())]:
    groups = {cat: subset[subset['Event_Category'] == cat]['Exp_Ratio'].dropna()
              for cat in cat_order}
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    if len(groups) < 2:
        continue

    f_stat_val, p_val = stats.f_oneway(*groups.values())
    anova_rows.append({
        'scope': scope, 'f_stat': f_stat_val, 'p_value': p_val,
        **{f'mean_{k}': v.mean() for k, v in groups.items()},
        **{f'n_{k}': len(v) for k, v in groups.items()},
    })

    if len(groups) >= 3:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            all_vals   = np.concatenate(list(groups.values()))
            all_labels = np.concatenate([[k] * len(v) for k, v in groups.items()])
            tukey      = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
            tdf        = pd.DataFrame(data=tukey._results_table.data[1:],
                                      columns=tukey._results_table.data[0])
            tdf['scope'] = scope
            tukey_rows.append(tdf[['scope', 'group1', 'group2', 'meandiff', 'p-adj', 'reject']])
        except ImportError:
            pass

agg5 = pd.DataFrame(anova_rows)
agg5.to_csv(os.path.join(AGG_DIR, 'agg_anova.csv'), index=False, encoding='utf-8-sig')
print(f"     agg_anova → {len(agg5)}행 저장")

if tukey_rows:
    agg5t = pd.concat(tukey_rows, ignore_index=True)
    agg5t.to_csv(os.path.join(AGG_DIR, 'agg_tukey.csv'), index=False, encoding='utf-8-sig')
    print(f"     agg_tukey → {len(agg5t)}행 저장")


# ── 6. 요일 × 주차 × 구간 집계 ───────────────────────────────────────────
print("[6/8] agg_weekday 생성 중...")
day_map = {2: '수', 3: '목', 4: '금', 5: '토', 6: '일', 0: '월', 1: '화'}
melted['day_name'] = melted['DayOfWeek'].map(day_map)
agg6 = (melted.groupby(['segment', 'day_name', 'Week_Idx'])['Exp']
        .mean().reset_index())
agg6.columns = ['segment', 'day_name', 'week_idx', 'avg_exp']
agg6.to_csv(os.path.join(AGG_DIR, 'agg_weekday.csv'), index=False, encoding='utf-8-sig')
print(f"     → {len(agg6)}행 저장")


# ── 7. 직업별 Pre/Post 요약 ───────────────────────────────────────────────
print("[7/8] agg_job_summary 생성 중...")
df_job = df[(df['Pre_Valid_Days'] >= MIN_VALID_DAYS) & (df['Post_Valid_Days'] >= MIN_VALID_DAYS)].copy()
agg7 = (df_job.groupby(['job', 'segment'])
        .agg(n=('name', 'count'), pre_avg=('Pre_Avg', 'mean'), post_avg=('Post_Avg', 'mean'))
        .reset_index())
agg7['growth_rate'] = (agg7['post_avg'] - agg7['pre_avg']) / agg7['pre_avg'] * 100
agg7.to_csv(os.path.join(AGG_DIR, 'agg_job_summary.csv'), index=False, encoding='utf-8-sig')
print(f"     → {len(agg7)}행 저장")


# ── 8. 이벤트 기간 영향 분석 ──────────────────────────────────────────────
print("[8/8] agg_event_impact 생성 중...")
showcase_dt_e = pd.to_datetime(SHOWCASE_DATE)
post_cols_e   = [c for c in daily_cols
                 if pd.to_datetime(c.replace('Daily_', '')) - pd.Timedelta(days=1) > showcase_dt_e]

event_defs = []
if os.path.exists(EVENT_LOG_PATH):
    with open(EVENT_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '~' in line and ':' in line:
                date_range, rest = line.split(':', 1)
                parts = rest.split(':', 1)
                start_str, end_str = date_range.split('~')
                event_defs.append({
                    'start':      pd.to_datetime(start_str.strip()),
                    'end':        pd.to_datetime(end_str.strip()),
                    'event_name': parts[0].strip(),
                    'event_type': parts[1].strip() if len(parts) > 1 else '',
                })

agg8_rows = []
for ev in event_defs:
    # Daily_ 컬럼의 실제 활동일 = 컬럼날짜 - 1일
    ev_cols  = [c for c in post_cols_e
                if ev['start'] <= pd.to_datetime(c.replace('Daily_', '')) - pd.Timedelta(days=1) <= ev['end']]
    non_cols = [c for c in post_cols_e if c not in ev_cols]
    if not ev_cols or not non_cols:
        continue

    df_ev = df.copy()
    ev_valid  = df_ev[ev_cols].notna().sum(axis=1)
    non_valid = df_ev[non_cols].notna().sum(axis=1)
    df_ev = df_ev[(ev_valid >= MIN_VALID_DAYS) & (non_valid >= MIN_VALID_DAYS)]
    df_ev['event_avg']  = df_ev[ev_cols].mean(axis=1)
    df_ev['non_ev_avg'] = df_ev[non_cols].mean(axis=1)

    for seg in sorted(df_ev['segment'].dropna().unique()):
        sd = df_ev[df_ev['segment'] == seg].dropna(subset=['event_avg', 'non_ev_avg'])
        if len(sd) < 5:
            continue
        t, p = stats.ttest_rel(sd['non_ev_avg'], sd['event_avg'])
        agg8_rows.append({
            'event_name':    ev['event_name'],
            'event_type':    ev['event_type'],
            'event_start':   ev['start'].strftime('%Y-%m-%d'),
            'event_end':     ev['end'].strftime('%Y-%m-%d'),
            'segment':       seg,
            'n':             len(sd),
            'non_event_avg': sd['non_ev_avg'].mean(),
            'event_avg':     sd['event_avg'].mean(),
            'growth_rate':   (sd['event_avg'].mean() - sd['non_ev_avg'].mean()) / sd['non_ev_avg'].mean() * 100,
            't_stat':        t,
            'p_value':       p,
        })

agg8 = pd.DataFrame(agg8_rows)
agg8.to_csv(os.path.join(AGG_DIR, 'agg_event_impact.csv'), index=False, encoding='utf-8-sig')
print(f"     → {len(agg8)}행 저장")


# ── 결과 요약 ─────────────────────────────────────────────────────────────
total_bytes    = sum(os.path.getsize(os.path.join(AGG_DIR, f))
                     for f in os.listdir(AGG_DIR) if f.endswith('.csv'))
original_bytes = os.path.getsize(PROCESSED_PATH)

print(f"\n[완료] 저장 위치: {AGG_DIR}")
print(f"       집계 파일 합계 : {total_bytes / 1024:.1f} KB")
print(f"       원본 대비      : {total_bytes / original_bytes * 100:.2f}%  "
      f"({original_bytes / 1024 / 1024:.0f} MB → {total_bytes / 1024:.0f} KB)")
