import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# ================= CONFIG =================
KEY_PATH = "../../config/api.txt"
INPUT_PATH = "data/step2_ocids.csv"
OUTPUT_PATH = "data/step3_final_data.csv"

# [공식 문서 반영] 캐릭터 정보는 최근 7일 이내만 조회 가능합니다.
# 오늘(22일) 기준, 가장 확실한 데이터인 21일(어제)로 설정합니다.
TARGET_DATE = "2026-01-21"
DAILY_LIMIT = 950


# ==========================================

def load_api_key():
    try:
        with open(KEY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        print(f"❌ {KEY_PATH} 파일을 찾을 수 없습니다.")
        return None


def get_stats(api_key, ocid):
    # 공식 문서 URL 구조 준수
    url = f"https://open.api.nexon.com/maplestory/v1/character/stat?ocid={ocid}&date={TARGET_DATE}"
    headers = {"x-nxopen-api-key": api_key}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json(), "OK"
        else:
            # 넥슨 API가 제공하는 상세 에러 코드(OPENAPIXXXXX)를 파싱합니다.
            error_msg = res.json().get('error', {}).get('name', f"HTTP_{res.status_code}")
            return None, error_msg
    except Exception as e:
        return None, type(e).__name__


def parse_all_stats(json_data):
    if not json_data or 'final_stat' not in json_data: return None
    result = {
        'character_class': json_data.get('character_class'),
        'character_level': json_data.get('character_level'),
    }
    # 모든 스탯을 동적으로 수집 (작성자님 요청 반영)
    for item in json_data['final_stat']:
        name = item['stat_name']
        val = item['stat_value']
        try:
            if isinstance(val, str):
                result[name] = float(val.replace(',', '').replace('%', ''))
            else:
                result[name] = val
        except:
            result[name] = val
    return result


if __name__ == "__main__":
    api_key = load_api_key()
    if not api_key: exit()

    if not os.path.exists(INPUT_PATH):
        print(f"❌ {INPUT_PATH}가 없습니다.")
        exit()

    df_ocids = pd.read_csv(INPUT_PATH)

    # 이어하기 체크
    if os.path.exists(OUTPUT_PATH):
        try:
            df_existing = pd.read_csv(OUTPUT_PATH)
            processed_ocids = set(df_existing['ocid'].astype(str))
            print(f"🔄 이어하기: {len(processed_ocids)}명 완료됨.")
        except:
            processed_ocids = set()
    else:
        processed_ocids = set()

    df_ocids['ocid'] = df_ocids['ocid'].astype(str)
    todo_list = df_ocids[~df_ocids['ocid'].isin(processed_ocids)].to_dict('records')

    print(f"🚀 총 {len(todo_list)}명 수집 시작 (조회일: {TARGET_DATE})")

    new_rows = []
    api_calls = 0

    for row in todo_list:
        if api_calls >= DAILY_LIMIT:
            print(f"\n🛑 오늘 API 한도({DAILY_LIMIT}) 도달!")
            break

        ocid = row['ocid']
        name = row['name']

        raw_data, status = get_stats(api_key, ocid)
        api_calls += 1

        if status == "OK":
            parsed = parse_all_stats(raw_data)
            if parsed:
                new_rows.append({**row, **parsed})
                print(f"[{api_calls}] ✅ {name} - 성공")
        else:
            # 7일 초과 시 OPENAPI00005 에러가 발생합니다.
            print(f"[{api_calls}] ❌ {name} - 실패 ({status})")

        # 중간 저장 (20명 단위)
        if len(new_rows) >= 20:
            current_df = pd.DataFrame(new_rows)
            is_first = not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0
            current_df.to_csv(OUTPUT_PATH, mode='a', header=is_first, index=False, encoding='utf-8-sig')
            new_rows = []

        time.sleep(0.05)

    if new_rows:
        current_df = pd.DataFrame(new_rows)
        is_first = not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0
        current_df.to_csv(OUTPUT_PATH, mode='a', header=is_first, index=False, encoding='utf-8-sig')

    print("\n✨ 작업 완료.")