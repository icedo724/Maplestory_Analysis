import pandas as pd
import numpy as np
import os
import sys

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(BASE_DIR, "data", "showcase")
PREPROCESSED_DIR = os.path.join(SAVE_DIR, "preprocessed")

INPUT_FILE = os.path.join(SAVE_DIR, "daily_tracking_lv.csv")
OUTPUT_FILE = os.path.join(PREPROCESSED_DIR, "daily_segment_processed.csv")

SHOWCASE_DATE = "2025-12-13"


# ==========================================

def preprocess_for_analysis():
    if not os.path.exists(INPUT_FILE):
        print("[오류] 원본 데이터 파일이 없습니다. 수집을 먼저 진행해주세요.")
        sys.exit()

    # 저장할 폴더가 없으면 생성
    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    print("[진행] 원본 데이터 로드 중...")
    df = pd.read_csv(INPUT_FILE)

    # 경험치 및 레벨 컬럼 추출
    exp_cols = sorted([c for c in df.columns if c.startswith('Exp_')])
    dates = [c.replace('Exp_', '') for c in exp_cols]

    if len(dates) < 2:
        print("[오류] 비교할 일자 데이터가 부족합니다 (최소 2일 필요).")
        sys.exit()

    print("[진행] 일일 경험치 획득량 계산 중...")
    daily_cols = []
    for i in range(1, len(dates)):
        prev_d = dates[i - 1]
        curr_d = dates[i]
        col_name = f'Daily_{curr_d}'

        # 일일 획득량 계산 및 하한선 처리
        df[col_name] = df[f'Exp_{curr_d}'] - df[f'Exp_{prev_d}']
        df.loc[df[col_name] < 0, col_name] = 0
        daily_cols.append(col_name)

    print("[진행] 레벨 구간 분류 중...")
    target_lv_col = f'Lv_{SHOWCASE_DATE}'
    if target_lv_col not in df.columns:
        lv_cols = sorted([c for c in df.columns if c.startswith('Lv_')])
        target_lv_col = lv_cols[-1]

    def get_segment(lv):
        if pd.isna(lv): return None
        if 285 <= lv < 290:
            return 'Lv.285~289'
        elif 290 <= lv < 295:
            return 'Lv.290~294'
        elif 295 <= lv <= 300:
            return 'Lv.295~299'
        return None

    df['segment'] = df[target_lv_col].apply(get_segment)
    df = df.dropna(subset=['segment'])

    print("[진행] 쇼케이스 전후 평균 계산 중...")
    pre_cols = [c for c in daily_cols if c.replace('Daily_', '') <= SHOWCASE_DATE]
    post_cols = [c for c in daily_cols if c.replace('Daily_', '') > SHOWCASE_DATE]

    df['Pre_Avg'] = df[pre_cols].mean(axis=1) if pre_cols else 0
    df['Post_Avg'] = df[post_cols].mean(axis=1) if post_cols else 0

    print("[저장] 분석용 데이터 저장 중...")
    cols_to_save = ['name', 'job', 'world', 'segment', 'Pre_Avg', 'Post_Avg'] + daily_cols
    df[cols_to_save].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"[완료] 전처리 성공! 총 {len(df)}명 데이터 변환 완료.")
    print(f"[정보] 저장 위치: {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_for_analysis()