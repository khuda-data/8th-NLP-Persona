# 실험 검증 및 증빙 자료 (Experiment Validation & Evidence)

## 🎯 목적

이 폴더는 **일반 데이터 분석과 차별화된 점**을 증빙하기 위한 분석 스크립트와 결과를 포함합니다.

### 일반 데이터 분석 vs 본 실험

| 구분 | 일반 데이터 분석 | 본 실험 (Time-Aware RAG) |
|:---|:---|:---|
| **접근법** | 통계적 집계 (평균, 상관계수) | 에이전트 기반 시뮬레이션 |
| **시간 처리** | 정적 스냅샷 | 동적 시간 가중치 (Time Decay) |
| **개인화** | 집단 평균 | 페르소나별 개인화된 의사결정 |
| **정보 반영** | 과거 데이터 전체 | 최신 정보 우선 (RAG + Time Weight) |
| **예측 방식** | 회귀/분류 모델 | 다중 에이전트 시뮬레이션 |

---

## 📊 증빙 분석 항목

### 1. Time Decay 효과 분석
**목적:** 시간 가중치가 실제로 최근 리뷰에 더 높은 영향력을 부여하는지 검증

**분석 내용:**
- 최근 리뷰 vs 오래된 리뷰의 선택 비율
- Time decay 파라미터에 따른 성능 변화
- Half-life에 따른 예측 정확도 변화

**스크립트:** `analyze_time_decay_effect.py`

---

### 2. 페르소나별 차이 분석
**목적:** 다양한 게이머 유형이 실제로 다른 의사결정을 하는지 검증

**분석 내용:**
- 페르소나 유형별 구매 의도 분포
- 페르소나별 시간에 따른 변화 패턴
- 페르소나별 리뷰 선호도 차이

**스크립트:** `analyze_persona_differences.py`

---

### 3. 시간에 따른 동적 변화 추적
**목적:** 시뮬레이션이 실제 여론 변화를 추적하는지 검증

**분석 내용:**
- 시계열 구매 의도 변화
- Ground Truth와의 시점별 상관관계
- 주요 이벤트 시점의 반응 분석

**스크립트:** `analyze_temporal_dynamics.py`

---

### 4. Team 2 vs Team 3 비교 분석
**목적:** Time decay가 실제로 성능 개선을 가져오는지 검증

**분석 내용:**
- 선택된 리뷰의 평균 작성일 비교
- 상관계수 차이 통계적 유의성 검증
- 예측 오차 분석

**스크립트:** `compare_team2_team3.py`

---

### 5. 일반 통계 분석 vs 시뮬레이션 비교
**목적:** 단순 통계 분석으로는 보이지 않는 패턴을 시뮬레이션이 포착하는지 증빙

**분석 내용:**
- 단순 평균 예측 vs 시뮬레이션 예측 비교
- 페르소나 다양성이 예측에 미치는 영향
- 집단 평균의 한계 vs 개인화 시뮬레이션의 장점

**스크립트:** `compare_statistical_vs_simulation.py`

---

## 🚀 실행 방법

### 전체 분석 실행
```bash
cd experiment_validation
python run_all_analyses.py
```

### 개별 분석 실행
```bash
# Time Decay 효과 분석
python analyze_time_decay_effect.py

# 페르소나별 차이 분석
python analyze_persona_differences.py

# 시간 동적 변화 추적
python analyze_temporal_dynamics.py

# Team 2 vs Team 3 비교
python compare_team2_team3.py

# 통계 분석 vs 시뮬레이션 비교
python compare_statistical_vs_simulation.py
```

---

## 📈 결과 해석

### 핵심 증빙 포인트

1. **Time Decay의 효과성**
   - 최근 리뷰가 더 높은 가중치를 받아 상위 랭킹에 등장
   - Time decay 적용 시 상관계수 개선

2. **페르소나 다양성의 중요성**
   - 단순 평균으로는 포착 불가능한 패턴 발견
   - 다양한 게이머 유형이 실제로 다른 반응 보임

3. **동적 정보 반영**
   - 정적 분석과 달리 시간에 따른 변화 추적 가능
   - 실제 여론 변화와의 높은 상관관계

4. **RAG의 효과**
   - 페르소나별 개인화된 정보 검색
   - 단순 통계보다 정확한 예측

---

## 📝 결과 저장 위치

- **그래프:** `experiment_validation/figures/`
- **통계 결과:** `experiment_validation/results/`
- **상세 리포트:** `experiment_validation/reports/`

