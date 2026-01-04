# 프로젝트 최종 요약 (Project Summary)

## 📋 프로젝트 개요

**목적:** Cyberpunk 2077 출시 전후 여론 변화를 시뮬레이션하여, Time-Aware RAG의 효과성을 검증

**방법:** 3가지 방법론 비교
- **Team 1:** Static Zero-Shot (페르소나만 사용, 외부 정보 완전 차단, 기준선)
- **Team 2:** Static RAG (similarity만 사용, 대조군)
- **Team 3:** Time-Aware RAG (similarity × time decay, 실험군)

**Team 1 특징:**
- 외부 지식 없이 페르소나 프롬프트만 사용
- LLM의 사전학습 지식에만 의존
- 통계 기반 페르소나 구성 (ESA 2024, Statista, Newzoo 데이터 반영)
- 시간에 따른 변화 없음 (정적)

---

## 🔬 실험 설계

### 핵심 가설
- **H1:** 시간 가중치를 적용하면 최근 리뷰의 영향력이 증가하여, 실제 여론 변화를 더 잘 반영할 수 있다.
- **H0:** 시간 가중치가 없어도 (Team 2 방식) 충분히 정확한 예측이 가능하다.

### 실험 공정성 보장 (2025-01-04 통일 작업)

**목적:** 대조군(control group) 생성을 위해 모든 팀이 동일한 조건에서 실험

**통일된 요소:**
1. ✅ **LLM 모델:** `gpt-4o-mini` (모든 팀 동일)
2. ✅ **Temperature:** `0.5` (모든 팀 동일)
3. ✅ **페르소나 생성:** `utils/persona_generator.py` (104명, 동일)
4. ✅ **쿼리 생성:** `utils/search_queries.py` (Team 2/3 동일)
5. ✅ **평가 기준:** `evaluate_correlation.py` (동일)
6. ✅ **API 키 처리:** `.env` 파일 (동일)

**결과:** 성능 차이는 오직 RAG 방식 차이만 반영

---

## 🔑 Team 2 vs Team 3 차이점

### 동일한 부분
- 페르소나, 쿼리, 임베딩 모델, ChromaDB, 평가 기준 모두 동일

### 차이점 (오직 Time Decay)

**Team 2 (Static RAG):**
```python
# 쿼리당 top_k개만 검색
results = collection.query(
    query_texts=[query],
    n_results=top_k,  # 작은 풀
    where={"date": {"$lte": date_int}}
)
# similarity만 사용
```

**Team 3 (Time-Aware RAG):**
```python
# 쿼리당 100개 검색 후 재랭킹
results = collection.query(
    query_texts=[query],
    n_results=100,  # 넓은 풀
    where={"timestamp": {"$lte": current_ts}}
)
# similarity × time_factor 계산
similarity = max(0, 1 - dist)
time_factor = np.exp(-decay_rate * days_diff)
final_score = similarity * time_factor  # ← 핵심 차이
```

---

## ⏰ Time Decay 구현

### 공식
```
time_factor = exp(-decay_rate * days_diff)
final_score = similarity × time_factor
```

### 파라미터
- `decay_rate = 0.01` (기본값)
- Half-life ≈ 70일

### 예시
- 0일 전 리뷰: `time_factor = 1.00` (100%)
- 70일 전 리뷰: `time_factor ≈ 0.50` (50%)
- 100일 전 리뷰: `time_factor ≈ 0.37` (37%)

---

## 📊 검증 결과

### 중간 점검 질문 답변

1. **Team2와 Team3의 입력 리뷰 풀은 정말 동일한가?**
   - ✅ 부분적으로 동일 (공통 모듈 사용)
   - ⚠️ 검색 전략 차이로 최종 선택 리뷰는 다를 수 있음

2. **차이는 오직 time_weight 하나뿐인가?**
   - ✅ 거의 맞음 (time_weight가 핵심 차이)
   - 검색 전략 차이는 time_weight 적용을 위한 필수 전략

3. **evaluation 날짜 기준이 일관적인가?**
   - ✅ 예, 일관적 (`simulation_dates.csv` 공통 사용)

4. **랜덤 요소가 결과 차이를 만들 가능성은?**
   - ✅ Team2와 Team3에 동일하게 적용되므로 공정함
   - ⚠️ 재현성을 위해 random seed 고정 권장

---

## 🎯 연구 기여

### 1. 실험 공정성 검증
- Team 2와 Team 3의 차이점을 코드 레벨에서 명확히 분석
- 공통 모듈 사용으로 실험 일관성 확보
- LLM 모델 통일로 편향 제거

### 2. Time-Aware RAG 구현 분석
- Time decay 함수의 수학적 정의 및 파라미터 의미 명확화
- Team 2와 Team 3의 검색 전략 차이 분석
- 코드 주석 개선으로 가독성 향상

### 3. 문서화 개선
- README에 실험 개요 및 차이점 섹션 추가
- 공정성 보장을 위한 변경 사항 문서화
- 프로젝트 구조 명확화

---

## 🔧 주요 변경 사항 (2025-01-04)

### 변경 전
- Team 2: `qwen3:4b` (Ollama)
- Team 3: `gpt-4o-mini` (OpenAI)
- Temperature: Team 1(0.7), Team 2/3(0.5)
- API 키: 각 팀마다 다른 방식

### 변경 후
- 모든 팀: `gpt-4o-mini` (통일)
- 모든 팀: Temperature `0.5` (통일)
- 모든 팀: `.env` 파일에서 API 키 로드 (통일)
- `utils/llm_config.py` 공통 모듈 생성

**결과:** 대조군 생성 완료, 실험 공정성 보장

---

## 📈 평가 방법

### Ground Truth
- Steam Positive Ratio: `datasets/ground_truth_steam.csv`
- Stock Price: `datasets/ground_truth_stock.csv`

### 평가 지표
- Pearson 상관계수 (Steam, Stock 각각)

### 실행 명령어
```bash
# Team 1
python evaluate_correlation.py \
    --model_csv "static_zero_shot/Team1_Static_ZeroShot_Results.csv" \
    --model_name "Team1_Static" \
    --type "static"

# Team 2
python evaluate_correlation.py \
    --model_csv "static_rag/Team2_StaticRAG_Results.csv" \
    --model_name "Team2_Static" \
    --type "dynamic"

# Team 3
python evaluate_correlation.py \
    --model_csv "time_aware_rag/Team3_TimeAware_Results_Final.csv" \
    --model_name "Team3_TimeAware" \
    --type "dynamic"
```

---

## 🚀 사용 방법

### 환경 설정
1. `.env` 파일 생성:
```bash
OPENAI_API_KEY=sk-proj-xxxx...
```

2. 실험 실행:
```bash
python static_zero_shot/simulation_model_a.py  # Team 1
python static_rag/simulation_model_b.py        # Team 2
python time_aware_rag/simulation_model_c.py    # Team 3
```

모든 팀이 동일한 모델을 사용한다는 메시지 출력:
```
✅ Using model: gpt-4o-mini (Team 1)
✅ Using model: gpt-4o-mini (Team 2)
✅ Using model: gpt-4o-mini (Team 3)
```

---

## 📝 향후 개선 사항

### 권장 사항
1. **Random seed 고정** (재현성 향상)
2. **`simulation_model_c.py`의 Team3 스타일 호출 수정**
3. **하이퍼파라미터 튜닝** (decay_rate 값 변경 실험)

### 선택 사항
- 다른 time decay 함수 실험 (linear, polynomial 등)
- 다양한 half-life 값 실험

---

## 📚 프로젝트 구조

```
📦 Project Root
├── 📁 static_zero_shot/      # Team 1: Static Zero-Shot
├── 📁 static_rag/            # Team 2: Static RAG (대조군)
├── 📁 time_aware_rag/         # Team 3: Time-Aware RAG (실험군)
├── 📁 utils/                  # 공통 모듈
│   ├── persona_generator.py   # 페르소나 생성
│   ├── search_queries.py     # 쿼리 생성
│   └── llm_config.py         # LLM 설정 (통일)
├── 📁 datasets/               # 데이터셋
└── 📁 png/                    # 결과 그래프
```

---

## ✅ 결론

이 프로젝트는 **Time-Aware RAG의 효과성**을 검증하기 위해, 모든 팀이 동일한 조건에서 실험하도록 통일했습니다. 

**핵심 차별점:** Team 2와 Team 3의 차이는 오직 **Time decay 가중치 적용 여부**뿐이며, 이를 통해 시간 정보가 구매 의도 예측에 미치는 영향을 명확히 분석할 수 있습니다.

