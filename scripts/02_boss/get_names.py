import requests
import pandas as pd
import time
import os

# ================= CONFIG =================
# 수집일
# 26/1/14, 26/1/21,
# 7일 단위 수집
TARGET_DATE = "2026-01-21"  # 오늘 수집 시점
MIN_LEVEL = 295
MAX_LEVEL = 300
HISTORY_WIDE_FILE = "data/step1_history_wide.csv"
NAMES_FILE = "data/step1_names.csv"  # 마스터 명단 (OCID용)


# ==========================================

def load_api_key():
    try:
        with open("../../config/api.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ [Error] api.txt 파일이 없습니다.")
        exit()


def get_ranking(api_key, page):
    url = f"https://open.api.nexon.com/maplestory/v1/ranking/overall?date={TARGET_DATE}&world_type=0&page={page}"
    headers = {"x-nxopen-api-key": api_key}
    return requests.get(url, headers=headers)


if __name__ == "__main__":
    if not os.path.exists("../../data"): os.makedirs("../../data")
    api_key = load_api_key()

    # 1. 기존 데이터 로드 (Wide Format)
    if os.path.exists(HISTORY_WIDE_FILE):
        df_history = pd.read_csv(HISTORY_WIDE_FILE)
        print(f"🔄 기존 히스토리 로드 완료: {len(df_history)}명 기록 중")
    else:
        # 파일이 없으면 이름과 직업을 기준으로 빈 데이터프레임 생성
        df_history = pd.DataFrame(columns=['name', 'job'])
        print("🆕 새로운 히스토리 파일을 생성합니다.")

    current_data = []  # 오늘 수집된 데이터 담기
    page = 1

    print(f"🚀 1단계: 와이드 포맷 수집 시작 (기준 열: {TARGET_DATE})")

    while True:
        res = get_ranking(api_key, page)
        if res.status_code != 200: break

        rankings = res.json().get('ranking', [])
        if not rankings: break

        max_lv = rankings[0]['character_level']
        min_lv = rankings[-1]['character_level']
        print(f"🔍 Page {page} 스캔... (Lv.{max_lv} ~ {min_lv})", end='\r')

        if max_lv < MIN_LEVEL: break

        for user in rankings:
            lv = user['character_level']
            name = user['character_name']
            job = user['class_name']

            if MIN_LEVEL <= lv <= MAX_LEVEL:
                current_data.append({'name': name, 'job': job, TARGET_DATE: lv})

        page += 1
        time.sleep(0.05)

    # 2. 데이터 병합 (Merge)
    df_today = pd.DataFrame(current_data)

    if not df_today.empty:
        # 기존 데이터와 오늘 데이터를 'name'과 'job'을 기준으로 합칩니다.
        # how='outer'를 써야 기존에 없던 유저(신규 진입자)도 행으로 추가됩니다.
        df_final = pd.merge(df_history, df_today, on=['name', 'job'], how='outer')

        # 3. 저장
        df_final.to_csv(HISTORY_WIDE_FILE, index=False, encoding='utf-8-sig')

        # 4. OCID용 마스터 명단도 업데이트 (중복 없이 이름만 관리)
        df_master = df_final[['name', 'job']].copy()
        # 기존 명단이 있다면 레벨 정보를 위해 원래 쓰던 names_file 형식을 유지해도 좋지만,
        # 여기서는 히스토리에서 파생된 최신 명단으로 저장합니다.
        df_master.to_csv(NAMES_FILE, index=False, encoding='utf-8-sig')

        print(f"\n✅ 업데이트 완료!")
        print(f"📊 현재 총 유저 수: {len(df_final)}명")
        print(f"📅 새 열 추가됨: {TARGET_DATE}")
    else:
        print("\n⚠️ 수집된 데이터가 없습니다.")