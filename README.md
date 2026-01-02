# 🎮 Cyberpunk 2077 Purchase Intent Simulation (Multi-Agent RAG)

이 프로젝트는 '사이버펑크 2077'의 출시 전후 여론 변화를 시뮬레이션하기 위해, **3가지 다른 방법론(Team 1, 2, 3)** 을 비교 분석합니다.
모든 팀은 공통된 평가 스크립트와 **공통 페르소나 모듈**을 사용하여 실험의 일관성을 유지합니다.

---

## 📂 프로젝트 구조 (Directory Structure)

```bash
📦 Project Root
├── 📜 evaluate_correlation.py       # [공통] 평가 및 시각화 스크립트
├── 📜 analyze_ground_truth_steam.py # [공통] Ground Truth 생성 (Steam)
├── 📜 analyze_ground_truth_stock.py # [공통] Ground Truth 생성 (Stock)
│
├── 📁 datasets/                     # [Data] 데이터셋 저장소
│   ├── ground_truth_steam.csv       # (자동 생성) Steam 정답지
│   ├── ground_truth_stock.csv       # (자동 생성) 주가 정답지
│   └── Cyberpunk_2077_Steam_Reviews.csv # 원본 리뷰 데이터
│
├── 📁 png/                          # [Image] 결과 그래프 저장소
│
├── 📁 utils/                        # [Module] 공통 유틸리티
│   └── persona_generator.py         # 페르소나 생성기 (Team 1, 2, 3 공용)
│
├── 📁 static_zero_shot/             # [Team 1] 작업 공간
│   ├── simulation_model_a.py        # 팀 1 시뮬레이션 코드
│   └── Team1_Static_ZeroShot_Results.csv # 팀 1 결과
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

### 1. 페르소나 모듈 사용 가이드 (For Team 2 & 3)

모든 팀은 `utils` 폴더의 생성기를 사용하여 **동일한 에이전트 집단(104명)** 을 생성해야 합니다.

**💻 Python 코드 작성 예시:**

```python
import sys
import os

# 1. 상위 폴더(Project Root)를 경로에 추가 (utils를 불러오기 위함)
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.persona_generator import generate_balanced_personas

# 2. 페르소나 생성 (13명씩 8개 유형 = 총 104명)
personas = generate_balanced_personas(n_per_type=13)

# 3. 시뮬레이션 루프 예시 (Team 2/3용)
results = []
current_date = "2020-12-10"  # 시뮬레이션 시점

for persona in personas:
    # ... (RAG 검색 및 LLM 호출 로직) ...
    
    # 4. 결과 저장 (Dictionary 구성)
    results.append({
        "Agent_ID": persona.id,
        "Name": persona.name,
        "Persona_Type": persona.gamer_type_name_display, # (필수) 영어 유형 이름
        "Decision": "YES",           # (필수) YES or NO
        "Simulation_Date": current_date, # (필수) Team 2, 3는 날짜 꼭 넣어야 함!
        "Reasoning": "Bugs are terrible..."
    })

```

### 2. CSV 결과 파일 양식 (매우 중요 ⭐)

평가 스크립트가 인식할 수 있도록 반드시 아래 컬럼명을 포함해야 합니다.

| 컬럼명 | 필수 여부 | 설명 | 예시 값 |
| --- | --- | --- | --- |
| **`Agent_ID`** | 필수 | 에이전트 고유 ID | `ultimate_gamer_1` |
| **`Persona_Type`** | 필수 | 게이머 유형 | `The Ultimate Gamer` |
| **`Decision`** | **필수** | 구매 의사 (YES/NO 파싱용) | `YES`, `NO` |
| **`Simulation_Date`** | **Team 2, 3 필수** | 시뮬레이션 시점 (YYYY-MM-DD) | `2020-12-10` |
| `Reasoning` | 선택 | 판단 이유 | `Because of bugs...` |

> **주의:** Team 1(Static)은 시간 변화가 없으므로 `Simulation_Date` 컬럼이 없어도 됩니다. (평가 시 `--type static` 옵션 사용)

### 3. 환경 설정 (Environment)

루트 경로에 `.env` 파일을 생성하고 API Key를 설정하세요.

```bash
OPENAI_API_KEY=sk-proj-xxxx...

```

### 4. 데이터 준비 (Data Setup)

대용량 리뷰 데이터는 Git에 없으므로 아래 명령어로 다운로드하여 **`datasets` 폴더**에 위치시킵니다.

```bash
mkdir -p datasets
curl -L -o datasets/cyberpunk_reviews.zip [https://www.kaggle.com/api/v1/datasets/download/filas1212/cyberpunk-2077-steam-reviews-as-of-aug-8-2024](https://www.kaggle.com/api/v1/datasets/download/filas1212/cyberpunk-2077-steam-reviews-as-of-aug-8-2024)
unzip datasets/cyberpunk_reviews.zip -d datasets/

```

Ground Truth 생성 (결과는 `datasets/` 폴더로 이동 권장):

```bash
python analyze_ground_truth_steam.py
python analyze_ground_truth_stock.py
mv ground_truth_*.csv datasets/

```

---

## 📈 평가 스크립트 사용법 (Evaluation)

모든 팀은 루트 경로의 `evaluate_correlation.py`를 사용하여 자신의 모델을 평가합니다.

### ✅ Team 1: Static Zero-Shot (정보 없음)

시간 변수 없이 고정된 구매율을 평가합니다.

```bash
python evaluate_correlation.py \
    --model_csv "static_zero_shot/Team1_Static_ZeroShot_Results.csv" \
    --model_name "Team1_Static" \
    --type "static" \
    --steam_gt "datasets/ground_truth_steam.csv" \
    --stock_gt "datasets/ground_truth_stock.csv"

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
    --steam_gt "datasets/ground_truth_steam.csv" \
    --stock_gt "datasets/ground_truth_stock.csv"

```

* **옵션:** `--type dynamic` 필수.
* **예상 결과:** Team 2는 완만한 변화, Team 3는 실제 데이터(GT)와 높은 상관계수(급격한 변화)를 보여야 함.

