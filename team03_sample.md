# Team 3 결과 정리

---

- **3팀 실험 주제:** Time-Aware RAG를 사용하여 최신 리뷰를 우선시하면서 '사이버펑크 2077' 구매 의사를 결정할 수 있는지 검증하기
- **역할 (Role):** Time-Aware RAG (시간 가중치 적용)
- **핵심 가설:** 최신 리뷰에 더 높은 가중치를 부여하면, 시간에 따른 여론 변화를 더 정확하게 반영할 수 있을 것이다. 특히 게임의 품질이 개선되는 시점(패치 후)을 더 잘 포착할 수 있을 것이다.

---

## 2. 실험 설계 (Experiment Design)

### 🔹 시뮬레이션 환경

- **Model:** OpenAI `gpt-4o-mini`
- **Agents:** Newzoo 게이머 유형 기반 104명 (8개 유형 × 13명)
- **Method:** **Time-Aware RAG**
    - Vector DB에서 쿼리와의 유사도(Cosine Similarity)로 리뷰 검색
    - **시간 가중치(Time Decay) 적용:** `time_factor = exp(-decay_rate * days_diff)`
    - 최신 리뷰일수록 높은 점수: `final_score = similarity × time_factor`
    - 특정 시점 이전의 리뷰만 필터링 (Strict Date Filtering)

### 🔹 평가 지표 (Evaluation Metric)

- **Ground Truth (정답지):**
    1. Steam 일별 긍정 리뷰 비율 (7-day Moving Avg)
    2. CD Projekt Red 주가 (Stock Price)
- **Metric:** 피어슨 상관계수 (Pearson Correlation)

---

## 3. 실험 결과 (Results)

### 📊 전체 통계

- **총 결정 수:** 40개
- **시뮬레이션 날짜 수:** 4일
- **YES 결정:** 30개 (75.0%)
- **NO 결정:** 10개 (25.0%)

### 📊 페르소나별 결정 분포

- **The All-Round Enthusiast:** YES: 6, NO: 2
- **The Backseat Gamer:** YES: 3, NO: 1
- **The Cloud Gamer:** YES: 2, NO: 6
- **The Hardware Enthusiast:** YES: 4, NO: 0
- **The Time Filler:** YES: 15, NO: 1

### 📊 상관계수 (Correlation)

- **Steam 긍정 리뷰 비율과의 상관계수:** `0.9013`
- **주가와의 상관계수:** `0.1051`

### 📈 시간에 따른 구매 비율 변화

- **평균 구매 비율:** 0.750
- **최소 구매 비율:** 0.500
- **최대 구매 비율:** 1.000

---

## 4. Team 2와의 비교


- **Team 2 (Static RAG):** 상관계수 `N/A` (Steam), `N/A` (Stock)
- **Team 3 (Time-Aware RAG):** 상관계수 `0.9013` (Steam), `0.1051` (Stock)
- **개선도:** 시간 가중치 적용으로 최신 여론을 더 잘 반영할 수 있게 되었음

---

## 5. Time Decay 구현 상세

### 수식

```
time_factor = exp(-decay_rate * days_diff)
final_score = similarity × time_factor
```

- **decay_rate:** 0.01 (기본값)
- **days_diff:** 현재 시점과 리뷰 작성일의 차이 (일 단위)
- **의미:** 100일 전 리뷰는 약 37% 가중치, 200일 전 리뷰는 약 14% 가중치

---

## 6. 실험 로그 (Sample)

### 날짜별 결정 샘플

- **2020-12-10:** YES: 7, NO: 3 (비율: 0.70)
- **2020-12-15:** YES: 8, NO: 2 (비율: 0.80)
- **2021-02-01:** YES: 5, NO: 5 (비율: 0.50)
- **2023-10-01:** YES: 10, NO: 0 (비율: 1.00)

### 최종 결정 분포

```
YES    0.750
NO     0.250
```

---

*생성 시간: 2026-01-05 07:40:10*
