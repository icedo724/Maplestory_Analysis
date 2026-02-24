import pandas as pd
import os
import glob

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_DIR = os.path.join(BASE_DIR, "data", "showcase")
TEMP_DIR = os.path.join(SAVE_DIR, "temp")
OUTPUT_FILE = os.path.join(SAVE_DIR, "daily_tracking_lv.csv")
LOG_FILE = os.path.join(SAVE_DIR, "completed_log.txt")


# ==========================================

def check_progress():
    print("=" * 40)
    print("📊 메이플 데이터 수집 중간 점검 📊")
    print("=" * 40)

    # 1. 완료된 날짜(도장 찍힌 날짜) 확인
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            completed_dates = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ 완전히 끝난 날짜: 총 {len(completed_dates)}일")
        if completed_dates:
            print(f"   (가장 최근 완료일: {completed_dates[-1]})")
    else:
        print("✅ 완전히 끝난 날짜: 없음 (아직 첫 날짜 수집 중이거나 파일 없음)")

    print("-" * 40)

    # 2. 메인 파일 누적 유저 수 확인
    if os.path.exists(OUTPUT_FILE):
        try:
            # 메모리 절약을 위해 이름 컬럼만 읽어옴
            df_main = pd.read_csv(OUTPUT_FILE, usecols=['name'])
            print(f"📁 메인 파일 누적 유저: 총 {len(df_main):,}명 추적 중")
        except Exception as e:
            print(f"📁 메인 파일 접근 불가 (수집기가 저장 중일 수 있음): {e}")
    else:
        print("📁 메인 파일: 아직 생성되지 않음")

    print("-" * 40)

    # 3. 현재 실시간 수집 중인 Temp 파일 확인
    temp_files = glob.glob(os.path.join(TEMP_DIR, "partial_*.csv"))
    if temp_files:
        for t_file in temp_files:
            try:
                date_str = os.path.basename(t_file).replace("partial_", "").replace(".csv", "")
                df_temp = pd.read_csv(t_file, usecols=['name'])
                current_count = len(df_temp)
                current_page = (current_count // 200) + 1
                print(f"⏳ 현재 실시간 수집 중 ({date_str})")
                print(f"   -> {current_count:,}명 수집 완료 (약 {current_page}페이지 쯤 도달)")
            except Exception as e:
                print(f"⏳ 실시간 수집 상태 읽기 실패: {e}")
    else:
        print("⏳ 현재 임시 파일(Temp)이 없습니다. (수집기가 꺼져 있거나, 막 하루치를 끝낸 상태)")

    print("=" * 40)


if __name__ == "__main__":
    check_progress()