import os

current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(current_dir)

AGG_DIR          = os.path.join(BASE_DIR, "data", "processed", "showcase", "aggregated")
SURVIVAL_FILE    = os.path.join(BASE_DIR, "data", "processed", "survival", "survival_data.csv")
CLUSTER_FILE     = os.path.join(BASE_DIR, "data", "processed", "segmentation", "clustered_users.csv")
USER_DETAIL_FILE = os.path.join(BASE_DIR, "data", "raw", "user_detail.csv")

SHOWCASE_DATE = "2025-12-13"

DETAIL_COLS = frozenset([
    'name', 'world', 'tier', 'world_group', 'latest_level',
    'ocid', 'date_create', 'character_class', 'access_flag', 'union_level',
])

STAT_GROUPS = {
    '전투 핵심': ['최대 스탯공격력', '데미지', '보스 몬스터 데미지', '최종 데미지',
                  '방어율 무시', '크리티컬 확률', '크리티컬 데미지'],
    '성장 지표': ['스타포스', '아케인포스', '어센틱포스', '전투력'],
    '기본 스탯': ['STR', 'DEX', 'INT', 'LUK', 'HP'],
    '활동 보조': ['추가 경험치 획득', '버프 지속시간', '재사용 대기시간 감소 (%)',
                  '공격 속도', '이동속도'],
    '기타':     ['아이템 드롭률', '메소 획득량', '스탠스', '방어력', '마력', '공격력'],
}
