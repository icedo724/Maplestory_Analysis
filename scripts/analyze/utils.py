"""
공통 전처리 유틸리티
— showcase / segmentation / survival 스크립트가 공유하는 상수·함수 모음.

주의: 날짜 필터링 후 dates 리스트에 비연속 구간이 생기면
     compute_daily_exp 내부에서 해당 쌍의 diff 가 다일(多日) 증분이 된다.
     completed_log.txt 가 연속 날짜를 담고 있는 한 문제없으나,
     수집 공백이 있는 경우 후속 분석에서 이 점을 고려해야 한다.
"""

import os

import numpy as np
import pandas as pd

# ── 레벨별 요구 경험치 (레벨업 보정용) ─────────────────────────────────────
# 단일 정의. 수치 변경 시 이 파일 한 곳만 수정하면 된다.
LEVEL_REQ_EXP = {
    284:  49_263_453_722_414,
    285:  99_512_176_519_276,
    286: 109_463_394_171_203,
    287: 120_409_733_588_323,
    288: 132_450_706_947_155,
    289: 145_695_777_641_870,
    290: 294_305_470_836_577,
    291: 323_736_017_920_234,
    292: 356_109_619_712_257,
    293: 391_720_581_683_482,
    294: 430_892_639_851_830,
    295: 870_403_132_500_696,
    296: 957_443_445_750_765,
    297: 1_053_187_790_325_841,
    298: 1_158_506_569_358_425,
    299: 1_737_759_854_037_637,
}


# ── 세그먼트 분류 ──────────────────────────────────────────────────────────

def get_segment(lv):
    """레벨 → 분석 구간 문자열.  범위 외이거나 NaN 이면 None 반환."""
    if pd.isna(lv):
        return None
    lv = int(lv)
    if 285 <= lv <= 289: return 'Lv.285~289'
    if 290 <= lv <= 294: return 'Lv.290~294'
    if 295 <= lv <= 299: return 'Lv.295~299'
    if lv == 300:        return 'Lv.300'
    return None


# ── 완료 날짜 관리 ─────────────────────────────────────────────────────────

def load_completed_dates(log_file):
    """
    completed_log.txt → 수집이 완료된 날짜 set 반환.
    파일이 없으면 빈 set 반환 (전체 날짜 사용).
    """
    if not os.path.exists(log_file):
        print("   [주의] completed_log.txt 없음 → 날짜 필터 미적용 (전체 날짜 사용)")
        return set()
    with open(log_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def filter_completed_dates(dates, log_file):
    """
    날짜 문자열 리스트를 completed_log.txt 기준으로 필터링.

    - completed_log 없으면 원본 리스트 그대로 반환.
    - 제거된 날짜 수와 마지막 유효 날짜를 출력한다.
    - Exp_/Lv_ 컬럼 동기화는 호출부 책임.
    """
    completed = load_completed_dates(log_file)
    if not completed:
        return dates

    filtered = [d for d in dates if d in completed]
    removed  = len(dates) - len(filtered)
    if removed:
        tail = f" (마지막: {filtered[-1]})" if filtered else ""
        print(f"   [정보] 미완료 날짜 {removed}일 제거 → 사용 날짜: {len(filtered)}일{tail}")
    return filtered


# ── 일일 경험치 계산 ───────────────────────────────────────────────────────

def compute_daily_exp(df, dates):
    """
    연속 날짜 쌍으로 일일 경험치 증분 컬럼을 계산한다.

    처리 규칙
    ---------
    API 미갱신  : 95 %+ 값 동일 → 해당 일 제외 (daily_cols 에 미포함)
    레벨업 보정 : diff < 0 이고 레벨 상승 → (req_exp − prev_exp + curr_exp)
                   ※ 단일 레벨 상승(prev_lv → prev_lv+1) 가정.
                     하루에 2레벨 이상 오른 경우 중간 레벨 요구량이 누락되어
                     증가분이 소폭 과소 추정됨. Lv.285+ 에서는 극히 드묾.
    잔여 음수   : 타이밍 오차로 보고 0 처리
    NaN 유지    : 관측 전 기간(신규 진입 이전)은 NaN 그대로

    Parameters
    ----------
    df    : pd.DataFrame  Exp_{date}, Lv_{date} 컬럼 포함
    dates : list[str]     완료일 필터 적용 후 오름차순 날짜 문자열 목록

    Returns
    -------
    daily_dict    : dict[str, pd.Series]  {col_name: 증분 Series}
    daily_cols    : list[str]             제외 날짜를 뺀 컬럼명 순서
    levelup_total : int                   레벨업 보정 적용 건수
    """
    daily_dict    = {}
    daily_cols    = []
    levelup_total = 0

    for i in range(1, len(dates)):
        prev_d, curr_d = dates[i - 1], dates[i]
        col_name = f'Daily_{curr_d}'

        prev_exp = df[f'Exp_{prev_d}']
        curr_exp = df[f'Exp_{curr_d}']
        prev_lv  = df.get(f'Lv_{prev_d}')   # 컬럼 없으면 None
        curr_lv  = df.get(f'Lv_{curr_d}')

        # ── API 미갱신 감지 (95 %+ 동일 → 해당 날 제외) ─────────────────────
        both = pd.concat([prev_exp.rename('p'), curr_exp.rename('c')], axis=1).dropna()
        if len(both) > 100:
            freeze_rate = (both['p'] == both['c']).mean()
            if freeze_rate > 0.95:
                print(f"   [제외] {curr_d}: API 미갱신 ({freeze_rate * 100:.1f}% 동일) → Daily 제외")
                continue

        diff = curr_exp - prev_exp

        # ── 레벨업 보정 : diff < 0 이면서 레벨이 오른 경우 ───────────────────
        if prev_lv is not None and curr_lv is not None:
            levelup_mask = (
                (diff < 0)
                & (curr_lv > prev_lv)
                & prev_lv.notna()
                & curr_lv.notna()
            )
            if levelup_mask.sum() > 0:
                levelup_total += int(levelup_mask.sum())
                req_exp   = prev_lv.map(LEVEL_REQ_EXP)
                corrected = req_exp - prev_exp + curr_exp
                diff      = diff.where(~levelup_mask, other=corrected)

        # ── 잔여 음수(타이밍 오차) → 0 처리, NaN 유지 ───────────────────────
        neg_mask   = diff < 0
        remain_neg = int(neg_mask.fillna(False).sum())
        if remain_neg > 0:
            print(f"   [정보] {curr_d}: 타이밍 차이 {remain_neg}건 → 0 처리")

        daily_dict[col_name] = diff.where(~neg_mask.fillna(False), other=0)
        daily_cols.append(col_name)

    if levelup_total > 0:
        print(f"   [정보] 레벨업 보정 적용: 총 {levelup_total:,}건")

    return daily_dict, daily_cols, levelup_total
