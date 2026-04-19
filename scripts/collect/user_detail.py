import requests
import pandas as pd
import numpy as np
import time
import os
import sys
import urllib3
from datetime import datetime, timedelta

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= CONFIG =================
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR         = os.path.join(BASE_DIR, "data", "raw")
KEY_PATH        = os.path.join(BASE_DIR, "config", "api.txt")

TRACKING_FILE   = os.path.join(RAW_DIR, "daily_tracking_lv.csv")
SAMPLE_FILE     = os.path.join(RAW_DIR, "user_detail_sample.csv")
OCID_CACHE_FILE = os.path.join(RAW_DIR, "ocid_cache.csv")
OUTPUT_FILE     = os.path.join(RAW_DIR, "user_detail.csv")
STAT_FILE       = os.path.join(RAW_DIR, "user_stat.csv")  # 마이그레이션용 임시파일

DAILY_LIMIT_PER_KEY = 950
REQUEST_INTERVAL    = 0.3
SAMPLE_RATE         = 0.03
MIN_STRATUM_ALL     = 30
RANDOM_SEED         = 42
OCID_SAVE_INTERVAL  = 100
SAVE_INTERVAL       = 200

EXCLUDE_STATS = {
    '최소 스탯공격력',
    '무기 숙련도',
    '속성 내성 무시',
    '상태이상 추가 데미지',
    '소환수 지속시간 증가',
}

# 스탯 수집 여부 판별에 사용
DETAIL_COLS = frozenset([
    'name', 'world', 'tier', 'world_group', 'latest_level',
    'ocid', 'date_create', 'character_class', 'access_flag', 'union_level',
])

DATE_STR = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
BASE_URL = "https://open.api.nexon.com/maplestory/v1"
# ==========================================

WORLD_GROUPS = {
    '스카니아':  '스카니아',
    '루나':      '루나',
    '엘리시움':  '엘리시움',
    '크로아':    '크로아',
    '챌린저스':  '챌린저스',
    '챌린저스2': '챌린저스',
    '챌린저스3': '챌린저스',
    '챌린저스4': '챌린저스',
}


def get_world_group(world):
    return WORLD_GROUPS.get(world, '기타')


def get_tier(level):
    if 295 <= level <= 299: return 1
    if 290 <= level <= 294: return 2
    if 285 <= level <= 289: return 3
    return None


# ─── KeyManager ──────────────────────────────────────────────────────────────

class KeyManager:
    def __init__(self, key_file_path):
        self.keys = self._load_keys(key_file_path)
        self.current_idx = 0
        self.usage_map   = {k: 0 for k in self.keys}
        self.dead_keys   = set()

    def _load_keys(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                keys = [line.strip() for line in f if line.strip()]
            if not keys:
                raise ValueError("키 파일 비어있음")
            print(f"[설정] API 키 {len(keys)}개 로드")
            return keys
        except Exception as e:
            print(f"[오류] 키 로드 실패: {e}")
            sys.exit()

    def get_current_key(self):
        while self.current_idx < len(self.keys):
            curr = self.keys[self.current_idx]
            if curr not in self.dead_keys and self.usage_map.get(curr, 0) < DAILY_LIMIT_PER_KEY:
                return curr
            self.current_idx += 1
        return None

    def add_usage(self):
        k = self.get_current_key()
        if k:
            self.usage_map[k] += 1

    def mark_dead(self, reason=""):
        k = self.get_current_key()
        if k:
            self.dead_keys.add(k)
            print(f"\n   [폐기] {self.current_idx + 1}번 키 ({reason})")
            self.current_idx += 1

    def mark_exhausted(self):
        k = self.get_current_key()
        if k:
            self.usage_map[k] = DAILY_LIMIT_PER_KEY + 999
            print(f"\n   [만료] {self.current_idx + 1}번 키 한도 초과")
            self.current_idx += 1

    def total_remaining(self):
        return sum(
            max(0, DAILY_LIMIT_PER_KEY - self.usage_map.get(k, 0))
            for i, k in enumerate(self.keys)
            if i >= self.current_idx and k not in self.dead_keys
        )


# ─── API 호출 ─────────────────────────────────────────────────────────────────

def api_get(endpoint, params, key_manager):
    for _ in range(3):
        ck = key_manager.get_current_key()
        if not ck:
            return None
        try:
            res = requests.get(
                f"{BASE_URL}/{endpoint}",
                headers={"x-nxopen-api-key": ck},
                params=params,
                timeout=5,
                verify=False,
            )
        except Exception as e:
            key_manager.mark_dead(f"ConnectionError: {e}")
            continue

        key_manager.add_usage()

        if res.status_code == 200:
            return res.json()
        if res.status_code == 429:
            key_manager.mark_exhausted()
            continue
        if res.status_code in (401, 403):
            key_manager.mark_dead(f"Auth {res.status_code}")
            continue
        if res.status_code == 404:
            return None
        time.sleep(1)

    return None


def fetch_ocid(name, key_manager):
    data = api_get("id", {"character_name": name}, key_manager)
    time.sleep(REQUEST_INTERVAL)
    return data.get("ocid") if data else None


def fetch_basic(ocid, key_manager):
    data = api_get("character/basic", {"ocid": ocid, "date": DATE_STR}, key_manager)
    time.sleep(REQUEST_INTERVAL)
    if not data:
        return None
    return {
        "date_create":      data.get("character_date_create"),
        "character_class":  data.get("character_class"),
        "access_flag":      data.get("access_flag"),
    }


def fetch_union(ocid, key_manager):
    data = api_get("user/union", {"ocid": ocid, "date": DATE_STR}, key_manager)
    time.sleep(REQUEST_INTERVAL)
    return data.get("union_level") if data else None


def fetch_stat(ocid, key_manager):
    data = api_get("character/stat", {"ocid": ocid, "date": DATE_STR}, key_manager)
    time.sleep(REQUEST_INTERVAL)
    if not data:
        return {}
    return {
        s['stat_name']: s['stat_value']
        for s in data.get('final_stat', [])
        if s['stat_name'] not in EXCLUDE_STATS
    }


# ─── 샘플 목록 생성 (1회만 실행) ─────────────────────────────────────────────

def build_sample_list():
    print("[샘플] daily_tracking_lv.csv 로드 중...")
    df = pd.read_csv(TRACKING_FILE)

    lv_cols = sorted(
        [c for c in df.columns if c.startswith('Lv_')],
        key=lambda x: pd.to_datetime(x.replace('Lv_', ''))
    )
    df['latest_level'] = df[lv_cols].apply(
        lambda row: row.dropna().iloc[-1] if row.dropna().any() else np.nan, axis=1
    )
    df = df.dropna(subset=['latest_level'])
    df['latest_level'] = df['latest_level'].astype(int)
    df['tier']         = df['latest_level'].apply(get_tier)
    df = df.dropna(subset=['tier']).copy()
    df['tier']         = df['tier'].astype(int)
    df['world_group']  = df['world'].apply(get_world_group)

    def sample_stratum(x):
        if len(x) < MIN_STRATUM_ALL:
            return x
        return x.sample(frac=SAMPLE_RATE, random_state=RANDOM_SEED)

    sampled_idx = (
        df.groupby(['world', 'tier'], group_keys=False)
        .apply(sample_stratum, include_groups=False)
        .index
    )
    sample = df.loc[sampled_idx].reset_index(drop=True)
    sample = sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    sample = sample[['name', 'world', 'tier', 'world_group', 'latest_level']]
    sample.to_csv(SAMPLE_FILE, index=False, encoding='utf-8-sig')

    strat_summary = df.groupby(['world', 'tier']).size().rename('population')
    sampled_count = sample.groupby(['world', 'tier']).size().rename('sampled')
    summary = pd.concat([strat_summary, sampled_count], axis=1).fillna(0).astype(int)
    summary['rate(%)'] = (summary['sampled'] / summary['population'] * 100).round(1)
    print(summary.to_string())
    print(f"\n[샘플] 총 {len(sample):,}명 (모집단 {len(df):,}명 중 {len(sample)/len(df)*100:.1f}%)")
    print(f"       예상 API 호출 수: ~{len(sample) * 4:,}건 / 예상 소요일: ~{len(sample) * 4 // 2800 + 1}일")
    return sample


# ─── 수집 상태 판별 ───────────────────────────────────────────────────────────

def _has_stat(row: dict) -> bool:
    return any(
        col not in DETAIL_COLS and val is not None and not (isinstance(val, float) and np.isnan(val))
        for col, val in row.items()
    )


# ─── 메인 수집 루프 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(SAMPLE_FILE):
        sample_df = build_sample_list()
    else:
        sample_df = pd.read_csv(SAMPLE_FILE)
        print(f"[정보] 기존 샘플 목록 로드: {len(sample_df):,}명")

    rows_dict = {}
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        rows_dict = existing_df.where(pd.notna(existing_df), other=None).to_dict('records')
        rows_dict = {r['name']: r for r in rows_dict}
        print(f"[정보] 기수집 데이터 로드: {len(rows_dict):,}명")

    if os.path.exists(STAT_FILE):
        stat_df = pd.read_csv(STAT_FILE)
        print(f"[마이그레이션] user_stat.csv 병합 중 ({len(stat_df):,}건)...")
        for _, row in stat_df.iterrows():
            name = row['name']
            if name in rows_dict:
                rows_dict[name].update(row.to_dict())
            else:
                rows_dict[name] = row.to_dict()
        if rows_dict:
            pd.DataFrame(list(rows_dict.values())).to_csv(
                OUTPUT_FILE, index=False, encoding='utf-8-sig'
            )
        os.remove(STAT_FILE)
        print(f"   → 통합 완료. user_stat.csv 삭제")

    ocid_cache = {}
    if os.path.exists(OCID_CACHE_FILE):
        ocid_cache = pd.read_csv(OCID_CACHE_FILE).set_index('name')['ocid'].to_dict()
        print(f"[정보] OCID 캐시 로드: {len(ocid_cache):,}건")

    done_detail = {name for name, r in rows_dict.items() if r.get('ocid')}
    done_stat   = {name for name, r in rows_dict.items() if _has_stat(r)}
    done_all    = done_detail & done_stat

    targets = sample_df[~sample_df['name'].isin(done_all)].reset_index(drop=True)
    total   = len(targets)
    n_stat_only = len([n for n in targets['name'] if n in done_detail])

    print(f"\n[시작] 수집 대상: {total:,}명  (스탯만 필요: {n_stat_only:,}명)\n")

    if total == 0:
        print("[완료] 모든 유저 수집 완료.")
        sys.exit()

    key_manager   = KeyManager(KEY_PATH)
    ocid_modified = False
    save_counter  = 0

    try:
        for idx, (_, row) in enumerate(targets.iterrows()):
            ck = key_manager.get_current_key()
            if not ck:
                print("\n[종료] 오늘 API 한도 소진. 내일 재실행하세요.")
                break

            name         = row['name']
            need_detail  = name not in done_detail
            existing_row = rows_dict.get(name, {})

            if need_detail:
                if name in ocid_cache:
                    ocid = ocid_cache[name]
                else:
                    ocid = fetch_ocid(name, key_manager)
                    if ocid:
                        ocid_cache[name] = ocid
                        ocid_modified = True
            else:
                ocid = existing_row.get('ocid')

            if not ocid:
                print(f"\n   [스킵] {name}: OCID 없음")
                continue

            if need_detail:
                basic       = fetch_basic(ocid, key_manager)
                union_level = fetch_union(ocid, key_manager)
                new_row = {
                    'name':            name,
                    'world':           row['world'],
                    'tier':            row['tier'],
                    'world_group':     row['world_group'],
                    'latest_level':    row['latest_level'],
                    'ocid':            ocid,
                    'date_create':     basic.get('date_create')     if basic else None,
                    'character_class': basic.get('character_class') if basic else None,
                    'access_flag':     basic.get('access_flag')     if basic else None,
                    'union_level':     union_level,
                }
                existing_row.update(new_row)

            stat_dict = fetch_stat(ocid, key_manager)
            existing_row.update(stat_dict)
            rows_dict[name] = existing_row

            save_counter += 1
            if save_counter % SAVE_INTERVAL == 0:
                pd.DataFrame(list(rows_dict.values())).to_csv(
                    OUTPUT_FILE, index=False, encoding='utf-8-sig'
                )

            if ocid_modified and (idx + 1) % OCID_SAVE_INTERVAL == 0:
                pd.DataFrame(list(ocid_cache.items()), columns=['name', 'ocid']).to_csv(
                    OCID_CACHE_FILE, index=False, encoding='utf-8-sig'
                )
                ocid_modified = False

            remaining_keys = key_manager.total_remaining()
            print(f"   [{idx + 1:>6}/{total}] {name:<20} | 잔여 키 호출: {remaining_keys}", end='\r')

    except KeyboardInterrupt:
        print("\n[중단] 사용자 중단.")

    finally:
        if rows_dict:
            df_out = pd.DataFrame(list(rows_dict.values()))
            detail_first = [c for c in DETAIL_COLS if c in df_out.columns]
            stat_cols    = [c for c in df_out.columns if c not in DETAIL_COLS]
            df_out = df_out[detail_first + stat_cols]
            df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        if ocid_modified:
            pd.DataFrame(list(ocid_cache.items()), columns=['name', 'ocid']).to_csv(
                OCID_CACHE_FILE, index=False, encoding='utf-8-sig'
            )

        collected = len([n for n in rows_dict if _has_stat(rows_dict[n])])
        print(f"\n[완료] 전체 {len(rows_dict):,}명 / 스탯 수집 완료: {collected:,}명")
        print(f"       저장 위치: {OUTPUT_FILE}")
