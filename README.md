# 메이플스토리 쇼케이스 영향 분석

넥슨 오픈API로 수집한 실제 유저 데이터를 바탕으로, 2025년 12월 메이플스토리 쇼케이스(대규모 업데이트 예고 이벤트)가 일일 경험치 획득량에 미친 영향을 통계적으로 검증한 프로젝트입니다.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mapleexp.streamlit.app/)

---

## 핵심 결과

| 레벨 구간 | 증가율 | p-value |
|---|---|---|
| Lv.285~289 | **+77.4%** | < 0.001 |
| Lv.290~294 | **+92.9%** | < 0.001 |
| Lv.295~299 | **+31.5%** | < 0.001 |

| 선데이 이벤트 | 경험치 비율 중앙값 |
|---|---|
| 경타포스 | **181.6%** |
| 사냥 | **127.7%** |
| 사냥 외 | **96.9%** |

ANOVA F = 7,106.1, p < 0.001

---

## 분석 설계

RCT가 불가능한 실제 서비스 환경에서 **자연 실험(Natural Experiment)** 방식을 채택했습니다.

- **쇼케이스 기준일**: 2025.12.13
- **비교 구간**: 기준일 기준 완전 대칭 ±35일 (계절성 혼입 최소화)
- **검정 방법 1**: 대응표본 t-검정 — 동일 유저의 Pre/Post 평균 쌍 비교
- **검정 방법 2**: One-way ANOVA + Tukey HSD — 선데이 이벤트 유형별 효율 차이
- **레벨업 보정**: 레벨업 발생일 경험치를 잔여분으로 환산
- **유효 유저 조건**: Pre·Post 각 7일 이상 관측치 보유 (유령·이탈 유저 제외)

---

## 프로젝트 구조

```
scripts/01_showcase/
├── A_get_showcase_impact.py   # 넥슨 오픈API 데이터 수집
├── B_preprocessing.py         # 레벨업 보정 및 유저 필터링
├── C_export_agg.py            # 집계 파일 생성 (152MB → ~50KB)
├── D_analysis.py              # 통계 분석 결과 출력
└── E_showcase_dashboard.py    # Streamlit 대시보드

data/showcase/aggregated/
├── agg_daily_segment.csv      # 날짜 × 구간별 일평균 경험치
├── agg_segment_summary.csv    # 구간별 Pre/Post 요약 + t-검정
├── agg_sunday_events.csv      # 선데이 이벤트별 변화율
├── agg_sunday_box.csv         # 박스플롯용 분위수
├── agg_anova.csv              # ANOVA 결과
├── agg_tukey.csv              # Tukey HSD 사후 검정
├── agg_weekday.csv            # 요일 × 주차 × 구간 집계
└── agg_job_summary.csv        # 직업별 Pre/Post 요약
```

---

## 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. API 키 설정
echo "YOUR_NEXON_API_KEY" > config/api.txt

# 3. 파이프라인 순서대로 실행
python scripts/01_showcase/A_get_showcase_impact.py
python scripts/01_showcase/B_preprocessing.py
python scripts/01_showcase/C_export_agg.py

# 4. 대시보드 실행
streamlit run scripts/01_showcase/E_showcase_dashboard.py

# 5. 분석 결과 출력
python scripts/01_showcase/D_analysis.py
```

---

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 데이터 수집 | 넥슨 오픈API, requests |
| 데이터 처리 | pandas, numpy |
| 통계 분석 | scipy, statsmodels |
| 시각화 | Plotly, Streamlit |
| 배포 | Streamlit Cloud |

---

## 한계 및 개선 방향

- **혼입 변수**: 쇼케이스와 동시 적용된 다른 패치 요인을 완전히 분리하지 못함
- **대조군 부재**: 쇼케이스 미적용 집단이 존재하지 않는 구조적 한계
- **개선 방향**: 이중차분법(DiD) 적용, 이벤트 단위 세분화 분류, 복수 쇼케이스 종단 추적
