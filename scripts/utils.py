import os

import numpy as np
import pandas as pd

# 레벨업 보정용 요구 경험치
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


def get_segment(lv):
    # 범위 외 또는 NaN → None 반환
    if pd.isna(lv):
        return None
    lv = int(lv)
    if 285 <= lv <= 289: return 'Lv.285~289'
    if 290 <= lv <= 294: return 'Lv.290~294'
    if 295 <= lv <= 299: return 'Lv.295~299'
    if lv == 300:        return 'Lv.300'
    return None


def load_completed_dates(log_file):
    if not os.path.exists(log_file):
        print("   [주의] completed_log.txt 없음 → 날짜 필터 미적용 (전체 날짜 사용)")
        return set()
    with open(log_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def filter_completed_dates(dates, log_file):
    # Exp_/Lv_ 컬럼 동기화는 호출부 책임
    completed = load_completed_dates(log_file)
    if not completed:
        return dates

    filtered = [d for d in dates if d in completed]
    removed  = len(dates) - len(filtered)
    if removed:
        tail = f" (마지막: {filtered[-1]})" if filtered else ""
        print(f"   [정보] 미완료 날짜 {removed}일 제거 → 사용 날짜: {len(filtered)}일{tail}")
    return filtered


def compute_daily_exp(df, dates):
    # 레벨업 보정: diff<0 + 레벨상승 → req_exp−prev+curr (단일 레벨 가정; 2레벨↑ 시 소폭 과소)
    daily_dict    = {}
    daily_cols    = []
    levelup_total = 0

    for i in range(1, len(dates)):
        prev_d, curr_d = dates[i - 1], dates[i]
        col_name = f'Daily_{curr_d}'

        prev_exp = df[f'Exp_{prev_d}']
        curr_exp = df[f'Exp_{curr_d}']
        prev_lv  = df.get(f'Lv_{prev_d}')
        curr_lv  = df.get(f'Lv_{curr_d}')

        # API 미갱신 감지: 95%+ 동일 → 제외
        both = pd.concat([prev_exp.rename('p'), curr_exp.rename('c')], axis=1).dropna()
        if len(both) > 100:
            freeze_rate = (both['p'] == both['c']).mean()
            if freeze_rate > 0.95:
                print(f"   [제외] {curr_d}: API 미갱신 ({freeze_rate * 100:.1f}% 동일) → Daily 제외")
                continue

        diff = curr_exp - prev_exp

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

        # 잔여 음수(타이밍 오차) → 0
        neg_mask   = diff < 0
        remain_neg = int(neg_mask.fillna(False).sum())
        if remain_neg > 0:
            print(f"   [정보] {curr_d}: 타이밍 차이 {remain_neg}건 → 0 처리")

        daily_dict[col_name] = diff.where(~neg_mask.fillna(False), other=0)
        daily_cols.append(col_name)

    if levelup_total > 0:
        print(f"   [정보] 레벨업 보정 적용: 총 {levelup_total:,}건")

    return daily_dict, daily_cols, levelup_total
