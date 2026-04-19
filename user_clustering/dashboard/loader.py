import os
import pandas as pd
import streamlit as st
from config import SURVIVAL_FILE, CLUSTER_FILE, USER_DETAIL_FILE, DETAIL_COLS


@st.cache_data
def load_survival():
    if not os.path.exists(SURVIVAL_FILE):
        return None
    df = pd.read_csv(SURVIVAL_FILE).rename(columns={
        'event_flag':    'event',
        'duration_days': 'duration',
    })
    if os.path.exists(CLUSTER_FILE):
        clusters = pd.read_csv(CLUSTER_FILE)[['name', 'cluster']]
        df = df.merge(clusters, on='name', how='left')
        df['cluster'] = df['cluster'].astype('Int64')
    return df


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
