import os
import sys
import json
import pandas as pd
import time
from openai import OpenAI

# ---------------------------------------------------------------------------
# 경로 설정 (utils import 및 CSV 저장용)
# ---------------------------------------------------------------------------
# 현재 이 파일이 있는 경로 (/.../static_zero_shot)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 상위 경로 (/.../khuda_pro) -> utils를 찾기 위해 필요
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.persona_generator import generate_balanced_personas, Persona
from utils.llm_config import get_llm_client, TEMPERATURE

# 1. LLM 클라이언트 초기화 (공통 모듈 사용)
client, MODEL_NAME = get_llm_client()
print(f"✅ Using model: {MODEL_NAME} (Team 1)")

# =============================================================================
# 2. Prompt 생성
# =============================================================================

def create_static_zeroshot_prompt(persona: Persona) -> tuple:
    persona_desc = f"""You are a {persona.age} year old {persona.gender} named '{persona.name}'.
Occupation: {persona.occupation}
[Gamer Type: {persona.gamer_type_name_display}]
{persona.description}

[Traits]
- Spending Level: {persona.traits['spending_level']}
- Information Seeking: {persona.traits['information_seeking']}"""

    system_prompt = f"""[ROLE]
{persona_desc}

[INSTRUCTION]
Make a decision based SOLELY on your 'traits' and 'prior knowledge' without any external information (news, bugs, reviews, etc.).
Answer honestly based on your gamer persona.

[OUTPUT FORMAT]
You MUST respond in the following JSON format:
{{
    "decision": "YES" or "NO" (Purchase Intention),
    "reasoning": "A short reason (1-2 sentences)"
}}"""
    
    user_prompt = "Is 'Cyberpunk 2077' worth buying? Will you buy it?"
    
    return system_prompt, user_prompt

# =============================================================================
# 3. API Call
# =============================================================================

def call_llm(system_prompt: str, user_prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

# =============================================================================
# 4. Main Execution
# =============================================================================

def run_experiment_a_modular(n_agents: int = 100):
    print("=" * 70)
    print(f"Task 1: Static Zero-Shot (Modularized English Version)")
    print("=" * 70)
    
    personas = generate_balanced_personas(n_per_type=13) 
    results = []
    
    for i, persona in enumerate(personas):
        system_prompt, user_prompt = create_static_zeroshot_prompt(persona)
        
        print(f"[{i+1}/{len(personas)}] {persona.gamer_type_name_display}...", end=" ")
        response = call_llm(system_prompt, user_prompt)
        
        if response:
            decision = response.get("decision", "NO").upper()
            reason = response.get("reasoning", "")
            print(f"-> {decision}")
            
            results.append({
                "Agent_ID": persona.id,
                "Name": persona.name,
                "Gender": persona.gender,
                "Age_Group": persona.age_group,
                "Persona_Type": persona.gamer_type_name_display,
                "Decision": decision,
                "Reasoning": reason,
                "System_Prompt": system_prompt
            })
        time.sleep(0.5)
        
    # 결과 저장
    df = pd.DataFrame(results)

    output_path = os.path.join(current_dir, "Team1_Static_ZeroShot_Results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print(f"결과 저장 경로: {output_path}")
    print(df['Decision'].value_counts(normalize=True))
    print("=" * 70)

if __name__ == "__main__":
    run_experiment_a_modular()

