# 📂 Team 2: Static RAG Implementation

이 디렉토리는 Team 2의 **Static RAG (Retrieval-Augmented Generation)** 시뮬레이션 코드를 포함하고 있습니다.
Team 3 (Time-Aware)와의 변인 통제를 위해 **공통 프롬프트, 공통 쿼리 선택 로직, 공통 페르소나**를 사용하되, **검색(Retrieval) 방식**에서만 차이를 두었습니다.

## 🛠️ 주요 구성 파일

### 1. `build_chroma_db.py`
*   **기능**: 원본 리뷰 데이터(`datasets/reviews/Cyberpunk_2077_Steam_Reviews.csv`)를 읽어 ChromaDB를 구축합니다.
*   **특징**:
    *   날짜를 정수형(`YYYYMMDD`)으로 변환하여 저장 (날짜 필터링 속도 최적화).
    *   임베딩 모델: `all-MiniLM-L6-v2` (SentenceTransformer).
    *   `--test` 옵션으로 소규모 데이터(5000건) 테스트 빌드 가능.

### 2. `rag_modules.py`
*   **기능**: ChromaDB 연결 및 검색을 담당하는 모듈입니다.
*   **Static RAG 로직**:
    *   **Strict Date Filtering**: 시뮬레이션 현재 시점(`$lte`) 이전의 데이터만 필터링합니다.
    *   **Similarity Search**: 시간 감쇠(Time-Decay) 없이, 쿼리와의 유사도(Cosine Similarity)가 높은 순서대로 결과를 반환합니다.
    *   **Output Format**: Team 3와 동일하게 `"- [Date] Review..."` 형식의 문자열 리스트를 반환합니다.

### 3. `simulation_model_b.py`
*   **기능**: 실제 여론 변화 시뮬레이션을 수행하는 메인 스크립트입니다.
*   **프로세스**:
    1.  `datasets/simulation_dates.csv`에서 시뮬레이션 날짜를 로드합니다 (Dec 2020 ~ Aug 2024).
    2.  `utils/persona_generator.py`를 통해 104명의 에이전트(8개 게이머 유형)를 생성합니다.
    3.  각 날짜별로 에이전트가 쿼리를 수행하고(4 Random + 1 General), RAG를 통해 리뷰를 검색합니다.
    4.  검색된 정보를 바탕으로 LLM(`gpt-4o-mini`)이 구매 의사결정(YES/NO)을 내립니다.
*   **결과 저장**: `Team2_StaticRAG_Results.csv`에 저장됩니다.

## 🚀 실행 방법

### 1. 데이터베이스 구축
```bash
# 전체 데이터 빌드 (시간 소요)
python static_rag/build_chroma_db.py

# 테스트용 빌드 (빠름)
python static_rag/build_chroma_db.py --test
```

### 2. 시뮬레이션 실행
```bash
# 전체 실행 (n_per_type=13, 총 104명)
python static_rag/simulation_model_b.py

# 테스트 실행 (n_per_type=1, 총 8명)
# script 내부의 __main__에서 n_per_type=1로 설정되어 있음
python static_rag/simulation_model_b.py
```

## ⚖️ Team 3 (Time-Aware)와의 차이점

| 구분 | Team 2 (Static RAG) | Team 3 (Time-Aware RAG) |
| :--- | :--- | :--- |
| **Search Logic** | **Similarity Only** | Similarity * Time-Decay Score |
| **Ranking** | 유사도 높은 순 | 최신성 + 유사도 결합 점수 순 |
| **Recency** | 필터링만 적용 (과거 데이터도 유사하면 상위 노출) | 최신 데이터 가중치 부여 |

***
*작성일: 2026-01-03*
