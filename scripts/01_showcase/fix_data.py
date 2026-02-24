import requests
import pandas as pd
import time
import os
import shutil
from datetime import datetime

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(BASE_DIR, "data", "showcase")
RAW_FILE = os.path.join(SAVE_DIR, "daily_tracking_lv.csv")
KEY_PATH = os.path.join(BASE_DIR, "config", "api.txt")

# 수리가 필요한 타겟 날짜
TARGET_DATES = ["2025-12-13", "2026-01-16"]


# ==========================================

def backup_original():
    if not os.path.exists(RAW_FILE):
        return False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = RAW_FILE.replace(".csv", f"_backup_{timestamp}.csv")
    shutil.copy2(RAW_FILE, backup_path)
    print(f"📦 [백업] 원본 파일이 안전하게 복사되었습니다: {os.path.basename(backup_path)}")
    return True


def get_api_key():
    with open(KEY_PATH, "r", encoding="utf-8") as f:
        return f.readline().strip()


def repair_logic():
    if not backup_original():
        print("❌ [오류] 원본 파일이 없어 수리를 중단합니다.")
        return

    df = pd.read_csv(RAW_FILE)
    api_key = get_api_key()
    url = "https://open.api.nexon.com/maplestory/v1/ranking/overall"

    for target_date in TARGET_DATES:
        print(f"\n🚀 [{target_date}] 데이터 재수집 및 수리 시작...")

        # 유저별로 API를 다시 찔러서 최신 데이터를 가져오는 대신,
        # 해당 날짜의 랭킹 페이지를 순회하며 우리 리스트에 있는 유저를 매칭합니다.
        # (이 방법이 API 호출 횟수를 획기적으로 줄여줍니다.)

        page = 1
        found_count = 0
        target_names = set(df['name'].tolist())
        repaired_data = {}

        while True:
            params = {"date": target_date, "world_type": 0, "page": page}
            res = requests.get(url, headers={"x-nxopen-api-key": api_key}, params=params)

            if res.status_code != 200:
                print(f"   ⚠️ [중단] {page}페이지에서 오류 발생 (코드: {res.status_code})")
                break

            data = res.json().get('ranking', [])
            if not data or data[-1]['character_level'] < 285:  # 최소 레벨 컷
                break

            for entry in data:
                name = entry['character_name']
                if name in target_names:
                    repaired_data[name] = {
                        f"Lv_{target_date}": entry['character_level'],
                        f"Exp_{target_date}": entry['character_exp']
                    }
                    found_count += 1

            print(f"   🔎 {page}페이지 스캔 중... (찾은 유저: {found_count}명)", end='\r')
            page += 1
            time.sleep(0.2)  # API 부하 방지

        # 데이터 반영
        if repaired_data:
            print(f"\n   ✅ [{target_date}] 총 {found_count}명의 데이터 수리 완료.")
            for name, values in repaired_data.items():
                for col, val in values.items():
                    df.loc[df['name'] == name, col] = val
        else:
            print(f"\n   ❌ [{target_date}] API에서 데이터를 가져오지 못했습니다.")

    # 최종 저장
    df.to_csv(RAW_FILE, index=False, encoding='utf-8-sig')
    print("\n✨ 모든 수리 작업이 완료되었습니다.")


if __name__ == "__main__":
    repair_logic()