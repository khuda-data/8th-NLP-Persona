#!/usr/bin/env python3
"""
Team 2: Static RAG Simulation (Async Version)
asyncio + aiohttp를 사용한 비동기 실행으로 속도 향상
"""

import os
import sys
import json
import pandas as pd
import random
import asyncio
from openai import AsyncOpenAI

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.persona_generator import generate_balanced_personas, Persona
from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY
from utils.llm_config import get_llm_client, TEMPERATURE
from static_rag.rag_modules import RAGRetriever

# 병렬 토크나이저 경고 억제
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 병렬 토크나이저 경고 억제
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Async OpenAI 클라이언트 초기화
def get_async_llm_client():
    """비동기 LLM 클라이언트 반환"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    
    client = AsyncOpenAI(api_key=api_key)
    model_name = "gpt-4o-mini"
    return client, model_name

client, MODEL_NAME = get_async_llm_client()
print(f"✅ Using model: {MODEL_NAME} (Team 2 - Async)")

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
# 3. 비동기 API 호출
# =============================================================================

async def call_llm_async(client: AsyncOpenAI, prompt: str, semaphore: asyncio.Semaphore) -> dict:
    """비동기 LLM 호출 (동시 실행 수 제한, 재시도 포함)"""
    async with semaphore:
        for attempt in range(3):
            try:
                res = await client.chat.completions.create(
                    model=MODEL_NAME, 
                    messages=[{"role": "system", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=TEMPERATURE,
                    timeout=60  # 초과 지연 방지
                )
                return json.loads(res.choices[0].message.content)
            except Exception as e:
                if attempt == 2:
                    print(f"[LLM Final Error] {e}", flush=True)
                    return {"decision": "NO", "reasoning": f"Error: {e}"}
                await asyncio.sleep(2 ** attempt)

# =============================================================================
# 4. 메인 실행 (비동기)
# =============================================================================

async def run_experiment_b_rag_async(n_per_type: int = 13, max_concurrent: int = 5):
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # 즉시 출력
    sys.stderr.reconfigure(line_buffering=True)
    
    # 진행/저장용 상태
    flush_every = 10  # 요청 10개마다 CSV 저장
    buffer = []
    written = 0
    yes_count = 0
    no_count = 0
    flush_lock = asyncio.Lock()
    
    print("=" * 70, flush=True)
    print(f"Task 2: Static RAG Simulation (Async)", flush=True)
    print("=" * 70, flush=True)

    # RAG 검색기 초기화
    print("Initializing RAG Retriever...")
    retriever = RAGRetriever()

    # 날짜 로드
    dates_df = pd.read_csv(SIMULATION_DATES_FILE)
    simulation_dates = dates_df['date'].tolist()
    
    # 에이전트 생성
    personas = generate_balanced_personas(n_per_type=n_per_type) 
    print(f"Generated {len(personas)} agents.")
    print(f"Total tasks: {len(simulation_dates)} dates × {len(personas)} agents = {len(simulation_dates) * len(personas)} decisions")
    print(f"Max concurrent requests: {max_concurrent}\n")

    # 동시 실행 수 제한을 위한 Semaphore
    semaphore = asyncio.Semaphore(max_concurrent)
    retrieval_lock = asyncio.Lock()  # ChromaDB는 thread-safe 아님
    
    total_steps = len(simulation_dates) * len(personas)
    
    async def process_agent_date(persona: Persona, date_str: str, step_num: int):
        """에이전트-날짜 조합 처리"""
        nonlocal completed
        if step_num <= 3:
            print(f"🟢 Start task {step_num}: {persona.id} @ {date_str}", flush=True)
        
        # 1. 쿼리 선정
        agent_queries = GAMER_TYPE_QUERIES.get(persona.gamer_type, [])
        selected_queries = []
        if len(agent_queries) >= 4:
            selected_queries = random.sample(agent_queries, 4)
        else:
            selected_queries = agent_queries
        selected_queries.append(GENERAL_QUERY)
        
        try:
            # 2~3. ChromaDB 검색 (직렬화)
            async with retrieval_lock:
                candidates = []
                for query in selected_queries:
                    reviews = retriever.retrieve_reviews(query, date_str, top_k=2)
                    candidates.extend(reviews)
                unique_candidates = list(set(candidates))
                final_docs = unique_candidates[:5]
                prompt = create_prompt(persona, date_str, final_docs)
            if step_num <= 3:
                print(f"🔍 Retrieved {len(prompt.splitlines())} lines of context for {persona.id} @ {date_str}", flush=True)
            
            # 4. 비동기 LLM 호출
            res = await asyncio.wait_for(call_llm_async(client, prompt, semaphore), timeout=120)
            
            decision = res.get("decision", "NO").upper()
            decision = "YES" if "YES" in decision else "NO"
            
            return {
                "Agent_ID": persona.id,
                "Name": persona.name,
                "Persona_Type": persona.gamer_type_name_display,
                "Decision": decision,
                "Simulation_Date": date_str,
                "Reasoning": res.get("reasoning", "")
            }
        except Exception as e:
            print(f"[Task Error] {persona.id} @ {date_str}: {e}", flush=True)
            return {
                "Agent_ID": persona.id,
                "Name": persona.name,
                "Persona_Type": persona.gamer_type_name_display,
                "Decision": "NO",
                "Simulation_Date": date_str,
                "Reasoning": f"Error: {e}"
            }
    
    # 모든 작업 생성
    tasks = []
    step_count = 0
    for date_str in simulation_dates:
        for persona in personas:
            step_count += 1
            tasks.append(process_agent_date(persona, date_str, step_count))
    
    print(f"🚀 Starting {len(tasks)} async tasks...\n")
    
    async def flush_buffer():
        nonlocal buffer, written
        if not buffer:
            return
        batch = buffer
        buffer = []
        def write_batch():
            header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
            pd.DataFrame(batch).to_csv(
                OUTPUT_FILE,
                mode="a",
                index=False,
                header=header,
                encoding="utf-8-sig"
            )
        async with flush_lock:
            await asyncio.to_thread(write_batch)
            written += len(batch)
            print(f"💾 Saved {written}/{total_steps} rows", flush=True)
    
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        if result["Decision"] == "YES":
            yes_count += 1
        else:
            no_count += 1
        buffer.append(result)
        
        total_done = written + len(buffer)
        if len(buffer) >= flush_every:
            await flush_buffer()
        if total_done % max(10, flush_every) == 0 or total_done == total_steps:
            progress_pct = total_done * 100 // total_steps
            print(f"✅ Completed: {total_done}/{total_steps} ({progress_pct}%)", flush=True)
    
    # 남은 버퍼 플러시
    await flush_buffer()
    
    # 최종 통계 출력
    total = yes_count + no_count
    decision_counts = {
        "YES": yes_count / total if total else 0,
        "NO": no_count / total if total else 0
    }
    
    print("\n" + "=" * 70)
    print("Decision")
    print(f"NO     {decision_counts.get('NO', 0):.3f}")
    print(f"YES    {decision_counts.get('YES', 0):.3f}")
    print("=" * 70)
    print(f"Simulation completed. Results saved to {OUTPUT_FILE}")
    print("=" * 70)

def main():
    """메인 함수"""
    asyncio.run(run_experiment_b_rag_async(n_per_type=13, max_concurrent=20))

if __name__ == "__main__":
    main()

