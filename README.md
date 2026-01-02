# 8th-NLP-Persona
---
# 🎮 Cyberpunk 2077 Purchase Intent Simulation (Multi-Agent RAG)

이 프로젝트는 '사이버펑크 2077'의 출시 전후 여론 변화를 시뮬레이션하기 위해, **3가지 다른 방법론(Team 1, 2, 3)**을 비교 분석합니다.
모든 팀은 공통된 평가 스크립트를 사용하여 실제 데이터(Steam Review, Stock Price)와의 상관계수를 측정합니다.

---

## 📂 프로젝트 구조 (Directory Structure)
각 팀은 지정된 폴더 내에서 작업하며, 결과 파일(`.csv`)을 해당 폴더에 저장해야 합니다.

```bash
📦 Project Root
├── 📜 evaluate_correlation.py       # [공통] 평가 및 시각화 스크립트
├── 📜 analyze_ground_truth_steam.py # [공통] Ground Truth 생성 (Steam)
├── 📜 analyze_ground_truth_stock.py # [공통] Ground Truth 생성 (Stock)
├── 📊 ground_truth_steam.csv        # (자동 생성) Steam 정답지
├── 📊 ground_truth_stock.csv        # (자동 생성) 주가 정답지
│
├── 📁 static_zero_shot/             # [Team 1] 작업 공간
│   ├── simulation_model_a_v3.py     # 팀 1 시뮬레이션 코드
│   └── Team1_Static_Results.csv     # 팀 1 결과
│
├── 📁 static_rag/                   # [Team 2] 작업 공간
│   ├── simulation_model_b.py        # 팀 2 시뮬레이션 코드
│   └── Team2_StaticRAG_Results.csv  # 팀 2 결과
│
└── 📁 time_aware_rag/               # [Team 3] 작업 공간
    ├── simulation_model_c.py        # 팀 3 시뮬레이션 코드
    └── Team3_TimeAware_Results.csv  # 팀 3 결과

```

---

## ⚡ 공통 작업 규칙 (Convention)

### 1. CSV 결과 파일 양식 (매우 중요 ⭐)

모든 팀의 시뮬레이션 결과 CSV는 **반드시 아래 컬럼명을 포함**해야 평가 스크립트가 작동합니다.

| 컬럼명 | 필수 여부 | 설명 | 예시 값 |
| --- | --- | --- | --- |
| **`Agent_ID`** | 필수 | 에이전트 고유 ID | `ultimate_gamer_1` |
| **`Persona_Type`** | 필수 | 게이머 유형 | `얼티밋 게이머` |
| **`Decision`** | **필수** | 구매 의사 (YES/NO 파싱용) | `YES`, `NO` |
| **`Simulation_Date`** | **Team 2, 3 필수** | 시뮬레이션 시점 (YYYY-MM-DD) | `2020-12-10` |
| `Reasoning` | 선택 | 판단 이유 | `버그가 많아서...` |

> **주의:** Team 1(Static)은 시간 변화가 없으므로 `Simulation_Date` 컬럼이 없어도 됩니다. (평가 시 `--type static` 옵션 사용)

### 2. 환경 설정 (Environment)

루트 경로에 `.env` 파일을 생성하고 API Key를 설정하세요.

```bash
OPENAI_API_KEY=sk-proj-xxxx...

```

### 3. 데이터 준비 (Data Setup)

대용량 리뷰 데이터는 Git에 없으므로 아래 명령어로 다운로드합니다.

```bash
curl -L -o cyberpunk_reviews.zip [https://www.kaggle.com/api/v1/datasets/download/filas1212/cyberpunk-2077-steam-reviews-as-of-aug-8-2024](https://www.kaggle.com/api/v1/datasets/download/filas1212/cyberpunk-2077-steam-reviews-as-of-aug-8-2024)
unzip cyberpunk_reviews.zip

```

Ground Truth 생성:

```bash
python analyze_ground_truth_steam.py
python analyze_ground_truth_stock.py

```

---

## 📈 평가 스크립트 사용법 (Evaluation)

모든 팀은 루트 경로의 `evaluate_correlation.py`를 사용하여 자신의 모델을 평가합니다.

### ✅ Team 1: Static Zero-Shot (정보 없음)

시간 변수 없이 고정된 구매율을 평가합니다.

```bash
python evaluate_correlation.py \
    --model_csv "static_zero_shot/Team1_Static_Results.csv" \
    --model_name "Team1_Static" \
    --type "static" \
    --steam_gt "ground_truth_steam.csv" \
    --stock_gt "ground_truth_stock.csv"

```

* **예상 결과:** 외부 정보가 없으므로 상관계수가 `NaN` (변화 없음)이어야 정상.

### ✅ Team 2 & 3: RAG Models (시계열 변화)

시간 흐름(`Simulation_Date`)에 따른 구매율 변화를 평가합니다.

```bash
# 예시: Team 3 실행 명령어
python evaluate_correlation.py \
    --model_csv "time_aware_rag/Team3_TimeAware_Results.csv" \
    --model_name "Team3_TimeAware" \
    --type "dynamic" \
    --steam_gt "ground_truth_steam.csv" \
    --stock_gt "ground_truth_stock.csv"

```

* **옵션:** `--type dynamic` 필수.
* **예상 결과:** Team 2는 완만한 변화, Team 3는 실제 데이터(GT)와 높은 상관계수(급격한 변화)를 보여야 함.

---

## 🚀 팀별 목표 (Goals)

1. **Team 1 (Static Zero-Shot):**
* LLM의 Prior Knowledge만 사용.
* **목표:** 외부 충격(뉴스, 여론)에 반응하지 못하는 '고정된 베이스라인'임을 증명.


2. **Team 2 (Static RAG):**
* 단순 유사도 기반 검색 (Cosine Similarity).
* **목표:** 과거와 현재 정보가 섞여서(Recency 무시) 여론 변화를 느리고 둔하게 반영함을 확인.


3. **Team 3 (Time-Aware RAG):**
* 시간 가중치(Time Decay) 적용 검색.
* **목표:** 최신 여론을 즉각 반영하여 실제 Steam/주가 그래프와 유사한 패턴(높은 상관계수) 달성.


