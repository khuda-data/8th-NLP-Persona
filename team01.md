# Team 1 결과 정리

---

- **1팀 실험 주제:** 외부 정보 없이 LLM 페르소나 자체의 성향만으로 '사이버펑크 2077' 구매 의사를 결정할 수 있는지 검증하기
- **역할 (Role):** Baseline Model (기준점 설정)
- **핵심 가설:** 외부 지식(뉴스, 리뷰 등)가 차단된 상태라면, 시점이 변해도 에이전트의 구매 의향은 변하지 않고 일정할 것이다.(시간에 따른 여론, 정보에 독립적)

---

## 2. 실험 설계 (Experiment Design)

### 🔹 시뮬레이션 환경

- **Model:** OpenAI `gpt-4o-mini`
- **Agents:** Newzoo 게이머 유형 기반 104명 (8개 유형 × 13명)
- **Method:** **Static Zero-Shot**
    - RAG(검색)를 전혀 사용하지 않음.
    - 오직 에이전트의 **내부 성향(Traits)**과 **사전 지식(Prior Knowledge)**만으로 판단.
    - 특정 시점(Date) 정보를 주입하지 않음

### 🔹 평가 지표 (Evaluation Metric)

- **Ground Truth (정답지):**
    1. Steam 일별 긍정 리뷰 비율 (7-day Moving Avg)
    2. CD Projekt Red 주가 (Stock Price)
- **Metric:** 피어슨 상관계수 (Pearson Correlation)

---

## 3. 실험 결과 (Results)

### 📊 전체 통계

- **총 에이전트 수:** 104명
- **YES 결정:** 39명 (37.5%)
- **NO 결정:** 65명 (62.5%)

### 📊 페르소나별 결정 분포

- **The All-Round Enthusiast:** YES: 13, NO: 0
- **The Backseat Gamer:** YES: 0, NO: 13
- **The Cloud Gamer:** YES: 0, NO: 13
- **The Conventional Player:** YES: 0, NO: 13
- **The Hardware Enthusiast:** YES: 13, NO: 0
- **The Popcorn Gamer:** YES: 0, NO: 13
- **The Time Filler:** YES: 0, NO: 13
- **The Ultimate Gamer:** YES: 13, NO: 0

### 📊 상관계수 (Correlation)

- **Steam 긍정 리뷰 비율과의 상관계수:** `NaN`
- **주가와의 상관계수:** `NaN`

*참고: Static 모델은 시간에 따라 변하지 않는 상수값이므로 분산이 0이 되어 상관계수가 정의되지 않습니다 (NaN).*

### 📉 수치 분석

- **구매 의향 패턴:** 모든 시뮬레이션 날짜에 대해 **동일한 구매 비율 유지 (Flat Red Line)**.
- **상관계수 (Correlation):** `NaN`
    - *이유:* 모델의 예측값(Model Ratio)이 시간의 흐름에도 전혀 변하지 않는 상수(Constant)이기 때문에, 분산이 0이 되어 상관계수 정의 불가능. 즉 시간에 따른 소비자 선호 경향을 표현하지 못한다.

---

## 4. 다음 실험과의 연계점

1. **Baseline 검증:**
    - 외부 정보(External Context)가 없으면, LLM 에이전트는 현실 세계의 이슈(버그 사태, 패치 등)에 전혀 반응하지 못한다.
    - 이는 곧 **현실의 여론 변화를 반영하려면 반드시 RAG(외부 지식)가 필요하다**는 사실을 말한다.
2. **페르소나 일관성 확인:**
    - *Ultimate Gamer*는 무조건 구매, *Time Filler*는 구매 거부 등 각 페르소나의 성격대로 일관된 답변을 내놓았다. 게이머 성격에 따른 프롬프트 엔지니어링이 정상적으로 작동한다.

---

## 5. 최종 평가

- Team 1의 결과를 Baseline(대조군)으로 설정.
- **Team 2 (Static RAG)** 및 **Team 3 (Time-Aware RAG)** 실험 결과와 비교하여, 외부 정보 주입 시 성능이 얼마나 향상되는지 정량적으로 측정한다.

---

## 6. 실험 로그 (Sample)

### 에이전트별 결정 샘플

[1/104] The Ultimate Gamer... -> YES
[2/104] The Ultimate Gamer... -> YES
[3/104] The Ultimate Gamer... -> YES
[4/104] The Ultimate Gamer... -> YES
[5/104] The Ultimate Gamer... -> YES
[6/104] The Ultimate Gamer... -> YES
[7/104] The Ultimate Gamer... -> YES
[8/104] The Ultimate Gamer... -> YES
[9/104] The Ultimate Gamer... -> YES
[10/104] The Ultimate Gamer... -> YES
[11/104] The Ultimate Gamer... -> YES
[12/104] The Ultimate Gamer... -> YES
[13/104] The Ultimate Gamer... -> YES
[14/104] The All-Round Enthusiast... -> YES
[15/104] The All-Round Enthusiast... -> YES
[16/104] The All-Round Enthusiast... -> YES
[17/104] The All-Round Enthusiast... -> YES
[18/104] The All-Round Enthusiast... -> YES
[19/104] The All-Round Enthusiast... -> YES
[20/104] The All-Round Enthusiast... -> YES

... (총 104개 에이전트)

### 최종 결정 분포

```
YES    0.375
NO     0.625
```

---

*생성 시간: 2026-01-05 07:40:10*
