import os
import sys
import json
import pandas as pd
import random
from openai import OpenAI
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.persona_generator import generate_balanced_personas, Persona
from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY
from static_rag.rag_modules import RAGRetriever

# 1. API키 및 환경 설정 (LLM Configuration)
load_dotenv()

# --- LLM 설정 (Configuration) ---
USE_OLLAMA = True # Local LLM 사용 여부
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3:4b"
OPENAI_MODEL = "gpt-4o-mini"

if USE_OLLAMA:
    print(f"🔹 Using Local LLM (Ollama): {OLLAMA_MODEL}")
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama" # Ollama는 api_key가 필요 없지만 클라이언트 호환성을 위해 더미 값 입력
    )
    MODEL_NAME = OLLAMA_MODEL
else:
    print(f"🔸 Using OpenAI API: {OPENAI_MODEL}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in .env")
        pass 
    client = OpenAI(api_key=api_key)
    MODEL_NAME = OPENAI_MODEL
# -------------------------------

OUTPUT_FILE = "static_rag/Team2_StaticRAG_Results.csv"
SIMULATION_DATES_FILE = "datasets/simulation_dates.csv"

# =============================================================================
# 2. 프롬프트 생성
# =============================================================================

def create_prompt(agent: Persona, current_date: str, context: list):
    context_str = "\n".join(context) if context else "(No reviews found.)"
    
    return f"""[ROLE]
You are a {agent.age} {agent.gender}.
Personality: '{agent.gamer_type_name_display}' ({agent.description})

[DATE]
Today is {current_date}.

[SEARCH RESULTS]
Reviews selected based on your interests and recentness (Time-Weighted):
{context_str}

[TASK]
Decide to buy 'Cyberpunk 2077' or not based strictly on the reviews above.
- The reviews are filtered by relevance and recency.
- Trust these reviews as the most important information available to you.

[OUTPUT]
JSON only:
{{
    "decision": "YES" or "NO",
    "reasoning": "Explain why based on the reviews."
}}
"""

# =============================================================================
# 3. API 호출
# =============================================================================

def call_llm(prompt: str) -> dict:
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        return json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        return {"decision": "NO", "reasoning": "Error"}

# =============================================================================
# 4. 메인 실행 (Main Execution)
# =============================================================================

def run_experiment_b_rag(n_per_type: int = 13):
    print("=" * 70)
    print(f"Task 2: Static RAG Simulation")
    print("=" * 70)

    # RAG 검색기 초기화
    print("Initializing RAG Retriever...")
    retriever = RAGRetriever()

    # 날짜 로드
    dates_df = pd.read_csv(SIMULATION_DATES_FILE)
    simulation_dates = dates_df['date'].tolist()
    
    # 에이전트 생성
    # README: "generate_balanced_personas(n_per_type=13)" (총 104명)
    personas = generate_balanced_personas(n_per_type=n_per_type) 
    print(f"Generated {len(personas)} agents.")

    results = []
    
    # 시뮬레이션 루프
    total_steps = len(simulation_dates) * len(personas)
    step_count = 0

    for date_str in simulation_dates:
        print(f"\n📅 Date: {date_str}")
        
        for persona in personas:
            step_count += 1
            # 1. 쿼리 선정 (Team 3 방식: 4개 랜덤 + 일반 쿼리)
            agent_queries = GAMER_TYPE_QUERIES.get(persona.gamer_type, [])
            selected_queries = []
            if len(agent_queries) >= 4:
                selected_queries = random.sample(agent_queries, 4)
            else:
                selected_queries = agent_queries # Fallback
            selected_queries.append(GENERAL_QUERY)
            
            # 2. 검색 (Team 2 정적 로직)
            # 쿼리당 상위 k개를 검색하고 합침
            # Team 3는 쿼리당 100개를 검색 후 시간 감쇠(Time-Decay) 랭킹을 적용하지만,
            # Team 2는 유사도(Similarity) 기반 상위 k개를 검색
            
            candidates = []
            for query in selected_queries:
                # retrieve_reviews 함수는 "- [Date] text..." 형식을 반환
                reviews = retriever.retrieve_reviews(query, date_str, top_k=2)
                candidates.extend(reviews)
            
            # 중복 제거 (단순 집합 사용)
            unique_candidates = list(set(candidates))
            
            # 상위 5개 선택 (Team 3와 동일 개수)
            final_docs = unique_candidates[:5]
            
            # 3. 프롬프트 생성
            prompt = create_prompt(persona, date_str, final_docs)
            
            # 4. LLM 호출
            print(f"   [{step_count}/{total_steps}] Agent {persona.id}...", end=" ", flush=True)
            res = call_llm(prompt)
            
            decision = res.get("decision", "NO").upper()
            decision = "YES" if "YES" in decision else "NO"
            
            print(f"-> {decision}")
            
            results.append({
                "Agent_ID": persona.id,
                "Name": persona.name,
                "Persona_Type": persona.gamer_type_name_display,
                "Decision": decision,
                "Simulation_Date": date_str,
                "Reasoning": res.get("reasoning", "")
            })

    # 결과 저장
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print(f"Simulation completed. Results saved to {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    # 테스트 실행 (유형별 1명 생성)
    run_experiment_b_rag(n_per_type=13)
