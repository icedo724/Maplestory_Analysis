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
LOG_FILE    = os.path.join(SAVE_DIR, "completed_log.txt")  # 수집 완료 날짜 로그

SHOWCASE_DATE = "2025-12-13"
# ==========================================


def load_completed_dates():
    """
    completed_log.txt에서 수집이 완료된 날짜 목록을 읽어 반환.
    로그 파일이 없으면 빈 set 반환 → 전체 날짜 사용 (하위 호환).
    """
    if not os.path.exists(LOG_FILE):
        print("   [주의] completed_log.txt 없음 → 날짜 필터 미적용 (전체 날짜 사용)")
        return set()
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def get_segment(lv):
    """
    레벨 구간 분류.
    [수정] 300을 295~299에 포함시키던 라벨 오류 → 별도 구간으로 분리
    """
    if pd.isna(lv):
        return None
    lv = int(lv)
    if 285 <= lv <= 289:
        return 'Lv.285~289'
    elif 290 <= lv <= 294:
        return 'Lv.290~294'
    elif 295 <= lv <= 299:
        return 'Lv.295~299'
    elif lv == 300:
        return 'Lv.300'
    return None


def preprocess_for_analysis():
    if not os.path.exists(INPUT_FILE):
        print("[오류] 원본 데이터 파일이 없습니다. 수집을 먼저 진행해주세요.")
        sys.exit()

    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)

    print("[진행] 원본 데이터 로드 중...")
    df = pd.read_csv(INPUT_FILE)

    # [개선] 명시적 날짜 파싱 정렬 (문자열 정렬 의존 제거)
    exp_cols = sorted(
        [c for c in df.columns if c.startswith('Exp_')],
        key=lambda x: pd.to_datetime(x.replace('Exp_', ''))
    )
    lv_cols_sorted = sorted(
        [c for c in df.columns if c.startswith('Lv_')],
        key=lambda x: pd.to_datetime(x.replace('Lv_', ''))
    )
    dates = [c.replace('Exp_', '') for c in exp_cols]

    # ── 미완료 날짜 제외 ────────────────────────────────────────────────────
    # completed_log.txt에 기록된 날짜만 사용.
    # 수집이 중단된 날(임시 파일만 있고 로그 미기록)은 데이터가 불완전하므로 제외.
    # Daily 컬럼은 "전날 → 당일" 증분이므로, 전날이 미완료라면 당일 diff도 오염됨.
    # → 완료된 날짜 기준으로 dates를 필터링한 뒤 exp/lv 컬럼도 동기화.
    completed = load_completed_dates()
    if completed:
        before_dates = len(dates)
        dates = [d for d in dates if d in completed]
        removed_dates = before_dates - len(dates)
        if removed_dates:
            print(f"   [정보] 미완료 날짜 {removed_dates}일 제외 → 사용 날짜: {len(dates)}일")
            print(f"          마지막 완료일: {dates[-1]}")
        # exp/lv 컬럼도 동기화
        exp_cols       = [f'Exp_{d}' for d in dates]
        lv_cols_sorted = [f'Lv_{d}'  for d in dates]

    if len(dates) < 2:
        print("[오류] 비교할 일자 데이터가 부족합니다 (최소 2일 필요).")
        sys.exit()

    # 레벨업 보정: diff < 0 시 (요구치 - prev_exp) + curr_exp 로 재계산
    # 신규 진입 유저(prev_exp=NaN)는 NaN 유지

    # 레벨별 요구 경험치
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

    print("[진행] 일일 경험치 획득량 계산 중 (레벨업 보정 포함)...")

    daily_dict = {}
    daily_cols = []
    levelup_total = 0

    for i in range(1, len(dates)):
        prev_d = dates[i - 1]
        curr_d = dates[i]
        col_name = f'Daily_{curr_d}'

        prev_exp = df[f'Exp_{prev_d}']
        curr_exp = df[f'Exp_{curr_d}']
        prev_lv  = df.get(f'Lv_{prev_d}')
        curr_lv  = df.get(f'Lv_{curr_d}')

        # API 미갱신 감지: 전날과 당일 스냅샷이 95% 이상 동일하면 해당 날 제외
        # (prev는 다음 날 계산 기준으로 유지)
        both = pd.concat([prev_exp.rename('p'), curr_exp.rename('c')], axis=1).dropna()
        if len(both) > 100:
            freeze_rate = (both['p'] == both['c']).mean()
            if freeze_rate > 0.95:
                print(f"   [제외] {curr_d}: API 미갱신 ({freeze_rate*100:.1f}% 동일) → Daily 제외")
                continue

        diff = curr_exp - prev_exp

        # 레벨업 감지: diff < 0 이고 레벨이 올랐을 때
        if prev_lv is not None and curr_lv is not None:
            levelup_mask = (diff < 0) & (curr_lv > prev_lv) & prev_lv.notna() & curr_lv.notna()
            levelup_count = int(levelup_mask.sum())
            if levelup_count > 0:
                levelup_total += levelup_count
                # 보정: 전날 레벨 요구치 - prev_exp + curr_exp
                req_exp = prev_lv.map(LEVEL_REQ_EXP)
                corrected = req_exp - prev_exp + curr_exp
                diff = diff.where(~levelup_mask, other=corrected)
        else:
            levelup_mask = pd.Series(False, index=diff.index)

        # 잔여 음수 = 레벨업 없는데 음수 → 스냅샷 타이밍 차이 → 0 처리
        remain_neg = (diff < 0) & ~levelup_mask
        remain_neg_count = int(remain_neg.fillna(False).sum())
        if remain_neg_count > 0:
            print(f"   [정보] {curr_d}: 타이밍 차이 {remain_neg_count}건 → 0 처리")

        # 음수만 0, NaN(신규 진입 이전)은 NaN 유지
        neg_mask = diff < 0
        daily_dict[col_name] = diff.where(~neg_mask.fillna(False), other=0)
        daily_cols.append(col_name)

    if levelup_total > 0:
        print(f"   [정보] 레벨업 보정 적용: 총 {levelup_total:,}건")

    # 일괄 concat → fragmentation 방지
    df = pd.concat([df, pd.DataFrame(daily_dict, index=df.index)], axis=1)

    print("[진행] 레벨 구간 분류 중...")
    target_lv_col = f'Lv_{SHOWCASE_DATE}'
    if target_lv_col not in df.columns:
        target_lv_col = lv_cols_sorted[-1]
        print(f"   [정보] SHOWCASE_DATE 레벨 컬럼 없음 → 최신 컬럼 사용: {target_lv_col}")

    df['segment'] = df[target_lv_col].apply(get_segment)
    df = df.dropna(subset=['segment'])

    # 300레벨 유저 플래그 (유령 유저 필터 오제거 방지용)
    is_lv300 = (df[target_lv_col] == 300)
    lv300_count = is_lv300.sum()
    if lv300_count > 0:
        print(f"   [정보] 300레벨 유저 {lv300_count}명 감지 → 유령 유저 필터 제외 처리")

    print("[진행] 쇼케이스 전후 평균 계산 중...")
    # Daily_YYYY-MM-DD = 전날 → 당일 증분
    # Daily_2025-12-13 = 12일→13일 증분 → 쇼케이스 당일 활동 → Pre에 포함
    showcase_dt = pd.to_datetime(SHOWCASE_DATE)
    pre_cols  = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) <= showcase_dt]
    post_cols = [c for c in daily_cols if pd.to_datetime(c.replace('Daily_', '')) > showcase_dt]

    # 0(비접속)은 포함, NaN(수집 공백/신규 진입 이전)은 제외
    df['Pre_Avg']  = df[pre_cols].mean(axis=1)  if pre_cols  else np.nan
    df['Post_Avg'] = df[post_cols].mean(axis=1) if post_cols else np.nan

    # 유효 데이터 일수 — Pre_Valid_Days가 적은 유저 = 수집 기간 중간 편입 유저
    df['Pre_Valid_Days']  = df[pre_cols].notna().sum(axis=1)  if pre_cols  else 0
    df['Post_Valid_Days'] = df[post_cols].notna().sum(axis=1) if post_cols else 0

    # 필터 1: 전 기간 경험치 합 0 또는 전부 NaN 유저 제거 (300레벨 제외)
    daily_sum  = df[daily_cols].sum(axis=1, min_count=1)
    ghost_mask = (daily_sum == 0) & (~is_lv300)
    nan_mask   = daily_sum.isna()

    before = len(df)
    df = df[~ghost_mask & ~nan_mask]
    print(f"   [필터 1] 전 기간 경험치 0 유저 {before - len(df)}명 제거 → 잔여 {len(df)}명")

    # 필터 2: 14일 이상 연속 경험치 0 유저 제거 (이탈 유저, NaN은 미포함, 300레벨 제외)
    INACTIVE_THRESHOLD = 14

    def has_inactive_streak(row, threshold):
        """연속 0 구간이 threshold일 이상인지 확인 (NaN은 카운트 제외)"""
        streak = 0
        for v in row:
            if pd.isna(v):
                streak = 0        # NaN은 연속 카운트 리셋 (수집 공백은 이탈로 보지 않음)
            elif v == 0:
                streak += 1
                if streak >= threshold:
                    return True
            else:
                streak = 0
        return False

    # 300레벨 유저는 검사 제외 후 다시 합치기
    is_lv300_now = (df[target_lv_col] == 300)
    df_300  = df[is_lv300_now]
    df_rest = df[~is_lv300_now]

    inactive_mask = df_rest[daily_cols].apply(
        lambda row: has_inactive_streak(row, INACTIVE_THRESHOLD), axis=1
    )

    before = len(df)
    df_rest = df_rest[~inactive_mask]
    df = pd.concat([df_rest, df_300], ignore_index=True)
    print(f"   [필터 2] 14일+ 연속 비활성 유저 {before - len(df)}명 제거 → 잔여 {len(df)}명")

    print("[저장] 분석용 데이터 저장 중...")
    cols_to_save = (
        ['name', 'job', 'world', 'segment',
         'Pre_Avg', 'Post_Avg',
         'Pre_Valid_Days', 'Post_Valid_Days']
        + daily_cols
    )
    df[cols_to_save].to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n[완료] 전처리 성공! 총 {len(df)}명 데이터 변환 완료.")
    print(f"[정보] 저장 위치: {OUTPUT_FILE}")
    print(f"[정보] Pre  기간: {pre_cols[0] if pre_cols else 'N/A'} ~ {pre_cols[-1] if pre_cols else 'N/A'} ({len(pre_cols)}일)")
    print(f"[정보] Post 기간: {post_cols[0] if post_cols else 'N/A'} ~ {post_cols[-1] if post_cols else 'N/A'} ({len(post_cols)}일)")


if __name__ == "__main__":
    preprocess_for_analysis()