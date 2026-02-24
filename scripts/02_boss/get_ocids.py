import requests
import pandas as pd
import time
import os
import datetime

# ================= CONFIG =================
INPUT_FILE = "data/step1_names.csv"  # 1단계 결과물
OUTPUT_FILE = "data/step2_ocids.csv"  # 저장될 파일
DAILY_LIMIT = 950  # 안전하게 950회에서 끊기


# ==========================================

def load_api_key():
    try:
        with open("../../config/api.txt", "r") as f:
            return f.read().strip()
    except:
        print("❌ api.txt 파일을 찾을 수 없습니다.")
        exit()


def get_ocid(api_key, name):
    url = f"https://open.api.nexon.com/maplestory/v1/id?character_name={name}"
    headers = {"x-nxopen-api-key": api_key}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 429:  # Too Many Requests
            time.sleep(5)
            return get_ocid(api_key, name)
        return res.json().get('ocid')
    except:
        return None


if __name__ == "__main__":
    api_key = load_api_key()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} 파일이 없습니다. 1단계부터 실행해주세요.")
        exit()

    # 1. 명단 로드
    df_names = pd.read_csv(INPUT_FILE)
    total_targets = len(df_names)

    # 2. 이어하기 확인
    processed_names = set()
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        processed_names = set(df_existing['name'])
        print(f"🔄 이어하기: 총 {total_targets}명 중 {len(processed_names)}명 완료됨.")
    else:
        # 파일 헤더 생성
        pd.DataFrame(columns=['name', 'level', 'job', 'world_type', 'ocid']).to_csv(OUTPUT_FILE, index=False,
                                                                                    encoding='utf-8-sig')

    # 3. 남은 작업량 계산
    # (주의: world_type 등 기존 컬럼을 유지하기 위해 merge 대신 필터링 사용)
    # 아직 처리 안 된 사람들의 인덱스만 추출
    remain_indices = df_names.index[~df_names['name'].isin(processed_names)]
    print(f"📋 오늘 처리할 남은 인원: {len(remain_indices)}명")

    api_calls = 0
    new_rows = []

    print(f"🚀 2단계 시작 (일일 제한 {DAILY_LIMIT}회)")

    for idx in remain_indices:
        # 일일 제한 체크
        if api_calls >= DAILY_LIMIT:
            print(f"\n🛑 [STOP] 일일 제한({DAILY_LIMIT}) 도달! 내일 다시 실행하세요.")
            break

        row = df_names.loc[idx]
        name = row['name']

        print(f"[{api_calls + 1}/{DAILY_LIMIT}] OCID 조회: {name} (Lv.{row['level']})", end='\r')

        ocid = get_ocid(api_key, name)
        api_calls += 1

        if ocid:
            data = row.to_dict()
            data['ocid'] = ocid
            new_rows.append(data)

        time.sleep(0.02)  # API 부하 방지

        # 50명마다 자동 저장
        if len(new_rows) >= 50:
            pd.DataFrame(new_rows).to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
            new_rows = []

            # 남은 데이터 저장
    if new_rows:
        pd.DataFrame(new_rows).to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

    print(f"\n✅ 오늘의 작업 종료. (API 호출: {api_calls}회)")