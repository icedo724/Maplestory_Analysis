import os
import pandas as pd
import streamlit as st
from config import AGG_DIR


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
