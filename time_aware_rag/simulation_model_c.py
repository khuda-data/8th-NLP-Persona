import os
import sys
import json
import pandas as pd
import random
from openai import OpenAI

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.persona_generator import generate_balanced_personas, Persona
from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY
from utils.llm_config import get_llm_client, TEMPERATURE
from time_aware_rag.rag_modules import RAGRetriever

# 1. LLM 클라이언트 초기화 (공통 모듈 사용)
client, MODEL_NAME = get_llm_client()
print(f"✅ Using model: {MODEL_NAME} (Team 3)")

OUTPUT_FILE = "time_aware_rag/Team3_TimeAware_Results_Final.csv"
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
            temperature=TEMPERATURE
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
    print(f"Task 3: Time Aware Rag Simulation")
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
            
            # 2. 검색 (Team 3 Time-Aware 로직)
            # 쿼리당 100개를 검색 후 시간 감쇠(Time-Decay) 랭킹을 적용
            # Team 2와의 차이: similarity × time_factor로 재랭킹
            
            # Persona 객체에 search_queries 속성 추가 (rag_modules.py 호환)
            # 노트북의 ChromaEnsembleRetriever.retrieve_weighted() 로직과 동일
            class PersonaWithQueries:
                def __init__(self, persona, queries):
                    self.search_queries = queries
            
            persona_with_queries = PersonaWithQueries(persona, selected_queries)
            
            # Time-Aware RAG 검색 (similarity × time_factor)
            # 노트북의 retrieve_weighted() 메서드와 동일한 로직
            final_docs = retriever.retrieve_reviews(
                persona_with_queries,
                current_date_str=date_str,
                top_k_final=5,
                decay_rate=0.01  # Half-life ≈ 70일
            )
            
            # 3. 프롬프트 생성
            prompt = create_prompt(persona, date_str, final_docs)
            
            # 4. LLM 호출
            print(f"[{step_count}/{total_steps}] {persona.gamer_type_name_display}...", end=" ", flush=True)
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
    
    # 최종 통계 출력
    decision_counts = df['Decision'].value_counts(normalize=True)
    
    print("\n" + "=" * 70)
    print("Decision")
    print(f"NO     {decision_counts.get('NO', 0):.3f}")
    print(f"YES    {decision_counts.get('YES', 0):.3f}")
    print("=" * 70)
    print(f"Simulation completed. Results saved to {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    # 테스트 실행 (유형별 1명 생성)
    run_experiment_b_rag(n_per_type=13)
