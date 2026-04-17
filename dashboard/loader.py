import os
import pandas as pd
import streamlit as st
from config import AGG_DIR, SURVIVAL_FILE, CLUSTER_FILE, USER_DETAIL_FILE, DETAIL_COLS


@st.cache_data
def load_agg():
    def p(name): return os.path.join(AGG_DIR, name)
    daily        = pd.read_csv(p('agg_daily_segment.csv'),  parse_dates=['date'])
    summary      = pd.read_csv(p('agg_segment_summary.csv'))
    sun_ev       = pd.read_csv(p('agg_sunday_events.csv'),  parse_dates=['Date'])
    sun_box      = pd.read_csv(p('agg_sunday_box.csv'))
    anova        = pd.read_csv(p('agg_anova.csv'))
    weekday      = pd.read_csv(p('agg_weekday.csv'))
    tukey_path   = p('agg_tukey.csv')
    tukey        = pd.read_csv(tukey_path) if os.path.exists(tukey_path) else pd.DataFrame()
    event_path   = p('agg_event_impact.csv')
    event_impact = pd.read_csv(event_path) if os.path.exists(event_path) else pd.DataFrame()
    return daily, summary, sun_ev, sun_box, anova, weekday, tukey, event_impact


@st.cache_data
def load_survival():
    if not os.path.exists(SURVIVAL_FILE):
        return None
    return pd.read_csv(SURVIVAL_FILE)


@st.cache_data
def load_cluster():
    if not os.path.exists(CLUSTER_FILE):
        return None
    return pd.read_csv(CLUSTER_FILE)


@st.cache_data
def load_user_stat():
    if not os.path.exists(USER_DETAIL_FILE):
        return None
    df = pd.read_csv(USER_DETAIL_FILE)
    stat_cols = [c for c in df.columns if c not in DETAIL_COLS]
    keep = ['name', 'world_group', 'tier'] + stat_cols
    return df[[c for c in keep if c in df.columns]]
