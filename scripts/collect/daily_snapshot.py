import requests
import pandas as pd
import time
import os
import sys
import urllib3
from datetime import datetime, timedelta

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(BASE_DIR, "data", "showcase")
TEMP_DIR = os.path.join(SAVE_DIR, "temp")  # 임시 저장소
OUTPUT_FILE = os.path.join(SAVE_DIR, "daily_tracking_lv.csv")
LOG_FILE = os.path.join(SAVE_DIR, "completed_log.txt")
KEY_PATH = os.path.join(BASE_DIR, "config", "api.txt")

MIN_ENTRY_LV = 285
MAX_ENTRY_LV = 300

DAILY_LIMIT_PER_KEY = 950
START_DATE = "2025-11-08"
REQUEST_INTERVAL = 0.2


# ==========================================

class KeyManager:
    def __init__(self, key_file_path):
        self.keys = self._load_keys(key_file_path)
        self.current_idx = 0
        self.usage_map = {k: 0 for k in self.keys}
        self.dead_keys = set()
        self.perform_health_check()

    def _load_keys(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                keys = [line.strip() for line in f.readlines() if line.strip()]
                if not keys: raise ValueError("키 파일 비어있음")
                print(f"[설정] 총 {len(keys)}개의 API 키 로드 완료")
                return keys
        except Exception as e:
            print(f"[오류] 키 파일 로드 실패: {e}")
            sys.exit()

    def perform_health_check(self):
        print("\n[점검] API 키 상태 확인 중...")
        url = "https://open.api.nexon.com/maplestory/v1/ranking/overall"
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        params = {"date": yesterday, "world_type": 0, "page": 1}

        for i, key in enumerate(self.keys):
            try:
                time.sleep(0.5)
                res = requests.get(url, headers={"x-nxopen-api-key": key}, params=params, verify=False, timeout=5)

                if res.status_code == 429:
                    print(f"   [제외] {i + 1}번 키: 한도 초과")
                    self.usage_map[key] = DAILY_LIMIT_PER_KEY + 9999
                elif res.status_code in [401, 403]:
                    print(f"   [폐기] {i + 1}번 키: 인증 실패")
                    self.dead_keys.add(key)
                elif res.status_code == 200:
                    print(f"   [정상] {i + 1}번 키: 사용 가능")
                else:
                    print(f"   [정보] {i + 1}번 키: 상태 코드 {res.status_code}")
            except Exception as e:
                print(f"   [오류] {i + 1}번 키: 연결 실패 ({e})")
        print("-" * 50)

    def get_current_key(self):
        while self.current_idx < len(self.keys):
            curr = self.keys[self.current_idx]
            if curr not in self.dead_keys and self.usage_map.get(curr, 0) < DAILY_LIMIT_PER_KEY:
                return curr
            self.switch_next()
        return None

    def add_usage(self):
        k = self.get_current_key()
        if k: self.usage_map[k] += 1

    def mark_dead(self, reason):
        k = self.get_current_key()
        if k:
            self.dead_keys.add(k)
            print(f"   [폐기] {self.current_idx + 1}번 키 ({reason})")
            self.switch_next()

    def mark_exhausted(self):
        k = self.get_current_key()
        if k:
            self.usage_map[k] = DAILY_LIMIT_PER_KEY + 999
            print(f"   [만료] {self.current_idx + 1}번 키 한도 초과")
            self.switch_next()

    def switch_next(self):
        if self.current_idx < len(self.keys):
            self.current_idx += 1


def get_ranking(api_key, date, page):
    url = "https://open.api.nexon.com/maplestory/v1/ranking/overall"
    params = {"date": date, "world_type": 0, "page": page}
    headers = {"x-nxopen-api-key": api_key}
    try:
        return requests.get(url, headers=headers, params=params, timeout=5, verify=False)
    except Exception as e:
        return e


def get_date_range(s, e):
    start = datetime.strptime(s, "%Y-%m-%d")
    end = datetime.strptime(e, "%Y-%m-%d")
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]


def load_completed_dates():
    if not os.path.exists(LOG_FILE): return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f.readlines() if line.strip()])


def mark_date_completed(date):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{date}\n")


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

    key_manager = KeyManager(KEY_PATH)

    completed_dates = load_completed_dates()
    print(f"[정보] 완료된 날짜 수: {len(completed_dates)}")

    tracked_names = set()
    if os.path.exists(OUTPUT_FILE):
        df_m = pd.read_csv(OUTPUT_FILE)
        tracked_names = set(df_m['name'].tolist())
        print(f"[정보] 기존 데이터 유저 수: {len(df_m)}")
    else:
        df_m = pd.DataFrame(columns=['name', 'job', 'world'])

    today = datetime.now().strftime("%Y-%m-%d")
    all_dates = get_date_range(START_DATE, today)
    target_dates = [d for d in all_dates if d not in completed_dates]

    if not target_dates:
        print("[정보] 모든 날짜의 수집이 완료되었습니다.")
        sys.exit()

    print(f"[시작] {target_dates[0]} ~ {target_dates[-1]} ({len(target_dates)}일치)")

    stop_all = False

    for target_date in target_dates:
        if stop_all: break

        # 임시 파일 이어받기
        temp_file_path = os.path.join(TEMP_DIR, f"partial_{target_date}.csv")
        page_file_path = os.path.join(TEMP_DIR, f"partial_{target_date}_page.txt")
        daily_data = []
        daily_names = set()
        start_page = 1

        if os.path.exists(temp_file_path):
            try:
                df_temp = pd.read_csv(temp_file_path)
                daily_data = df_temp.to_dict('records')
                daily_names = {r['name'] for r in daily_data}
                collected_count = len(daily_data)
                if os.path.exists(page_file_path):
                    with open(page_file_path, 'r') as pf:
                        start_page = int(pf.read().strip()) + 1
                else:
                    start_page = 1
                print(f"[복구] {target_date} 데이터 {collected_count}명 발견 -> {start_page}페이지부터 이어하기")
            except Exception as e:
                print(f"[오류] 임시 파일 손상, 처음부터 시작: {e}")
                daily_data = []
                start_page = 1
                if os.path.exists(page_file_path):
                    os.remove(page_file_path)

        if not key_manager.get_current_key():
            print("[종료] 모든 키 소진됨")
            break

        print(f"\n[진행] {target_date} 수집 중 (시작 페이지: {start_page})...")

        page = start_page
        consecutive_429 = 0
        date_success = False

        while True:
            ck = key_manager.get_current_key()
            if not ck:
                print("[중단] 가용 키 없음")
                stop_all = True
                break

            res = get_ranking(ck, target_date, page)
            key_manager.add_usage()

            if isinstance(res, Exception):
                print(f"   [오류] 연결 실패: {res}")
                key_manager.mark_dead("ConnectionError")
                continue

            if res.status_code != 200:
                if res.status_code == 429:
                    consecutive_429 += 1
                    if consecutive_429 < 3:
                        print(f"   [대기] 429 감지 ({consecutive_429}/3)")
                        time.sleep(5)
                        continue
                    else:
                        key_manager.mark_exhausted()
                        consecutive_429 = 0
                        continue
                elif res.status_code in [401, 403]:
                    key_manager.mark_dead(f"Auth {res.status_code}")
                else:
                    print(f"   [오류] 서버 응답 {res.status_code}")
                    time.sleep(2)
                continue

            consecutive_429 = 0
            data = res.json().get('ranking', [])

            if not data:
                date_success = True
                break

            max_lv = data[0]['character_level']
            min_lv = data[-1]['character_level']

            curr_k_idx = key_manager.current_idx + 1
            print(f"   [P.{page}] Lv.{max_lv}~{min_lv} | 누적: {len(daily_data)} | 키: {curr_k_idx}번", end='\r')

            if max_lv < MIN_ENTRY_LV:
                print(f"\n   [완료] 최저 레벨 도달")
                date_success = True
                break

            for user in data:
                lv = user['character_level']
                nm = user['character_name']
                if ((MIN_ENTRY_LV <= lv < MAX_ENTRY_LV) or (nm in tracked_names)) and nm not in daily_names:
                    daily_data.append({
                        'name': nm, 'job': user['class_name'], 'world': user['world_name'],
                        f"Lv_{target_date}": lv, f"Exp_{target_date}": user['character_exp']
                    })
                    daily_names.add(nm)
                    tracked_names.add(nm)

            # 페이지 단위 임시 저장
            if daily_data:
                pd.DataFrame(daily_data).to_csv(temp_file_path, index=False, encoding='utf-8-sig')
                with open(page_file_path, 'w') as pf:
                    pf.write(str(page))

            page += 1
            time.sleep(REQUEST_INTERVAL)

        # 날짜 완주 시 처리
        if date_success:
            if daily_data:
                # 1. 메인 데이터와 병합
                df_new = pd.DataFrame(daily_data)

                # 동일 날짜 컬럼 덮어쓰기
                cols_to_drop = [c for c in df_m.columns if target_date in c]
                if cols_to_drop:
                    df_m.drop(columns=cols_to_drop, inplace=True)

                df_m = pd.merge(df_m, df_new, on=['name', 'job', 'world'], how='outer')
                df_m.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

                # 임시 파일 삭제 및 완료 로그 기록
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.path.exists(page_file_path):
                    os.remove(page_file_path)

                mark_date_completed(target_date)
                print(f"\n[성공] {target_date} 처리 완료 (총 {len(df_m)}명)")
            else:
                # 데이터가 없는 날 (점검일 등?)
                mark_date_completed(target_date)
                print(f"\n[성공] {target_date} 데이터 없음 (완료 처리)")
        else:
            print(f"\n[중단] {target_date} 키 소진으로 일시 정지 (임시 파일 저장됨)")