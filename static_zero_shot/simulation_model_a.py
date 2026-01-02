"""
=============================================================================
Task 1: Static Zero-Shot 실험 (Compatible Version)
=============================================================================
수정 사항:
1. 평가 스크립트(evaluate_correlation.py) 호환을 위한 CSV 컬럼명 매핑
2. API Key 보안 적용 (.env)
3. 모델명 수정 (gpt-4o-mini)
4. 에이전트 수 조정 (약 100명)
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
import time
import random
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# =============================================================================
# 1. 통계 상수 정의 (Source: ESA 2024, Statista, Newzoo) - 원본 유지
# =============================================================================

GENDER_DISTRIBUTION = {"Male": 0.54, "Female": 0.46}

AGE_DISTRIBUTION = {
    "18-19": 0.04, "20-29": 0.24, "30-39": 0.26,
    "40-49": 0.21, "50-59": 0.17, "60+": 0.08
}

GAMER_TYPE_DISTRIBUTION = {
    "ultimate_gamer": 0.13, "all_round_enthusiast": 0.09, "cloud_gamer": 0.19,
    "conventional_player": 0.04, "hardware_enthusiast": 0.09, "popcorn_gamer": 0.13,
    "backseat_gamer": 0.06, "time_filler": 0.27
}

# =============================================================================
# 2. Newzoo 기반 상세 게이머 유형 정의 - 원본 유지
# =============================================================================

GAMER_TYPES = {
    "ultimate_gamer": {
        "type_name": "The Ultimate Gamer",
        "type_name_kr": "얼티밋 게이머",
        "proportion": 0.13,
        "description": "게임에 돈과 시간을 아끼지 않는 열정적인 게이머",
        "traits": {"spending_level": "Very High", "time_investment": "20+ hours/week", "platform_preference": ["PC", "Console"], "purchase_timing": "Day-1 구매", "information_seeking": "리뷰 상관없이 구매", "brand_loyalty": "Very High"},
        "cyberpunk_tendency": "무조건 구매",
        "expected_score_range": (75, 95)
    },
    "all_round_enthusiast": {
        "type_name": "The All-Round Enthusiast",
        "type_name_kr": "올라운드 열정가",
        "proportion": 0.09,
        "description": "모든 장르를 즐기며 균형 잡힌 게임 생활을 추구",
        "traits": {"spending_level": "Medium-High", "time_investment": "10-15 hours/week", "platform_preference": ["PC", "Console", "Mobile"], "purchase_timing": "리뷰 확인 후 구매", "information_seeking": "리뷰 꼼꼼히 확인", "brand_loyalty": "Medium"},
        "cyberpunk_tendency": "평가 좋으면 구매",
        "expected_score_range": (50, 80)
    },
    "cloud_gamer": {
        "type_name": "The Cloud Gamer",
        "type_name_kr": "클라우드 게이머",
        "proportion": 0.19,
        "description": "고사양 PC 없이 스트리밍/할인 게임 위주로 플레이",
        "traits": {"spending_level": "Low-Medium", "time_investment": "5-10 hours/week", "platform_preference": ["Cloud Gaming"], "purchase_timing": "대폭 할인 시 구매", "information_seeking": "최적화 리뷰 확인", "brand_loyalty": "Low"},
        "cyberpunk_tendency": "최적화 나쁘면 안 삼",
        "expected_score_range": (20, 60)
    },
    "conventional_player": {
        "type_name": "The Conventional Player",
        "type_name_kr": "전통적 플레이어",
        "proportion": 0.04,
        "description": "익숙한 게임만 반복 플레이, 신작에 관심 없음",
        "traits": {"spending_level": "Very Low", "time_investment": "5-10 hours/week", "platform_preference": ["PC", "Console"], "purchase_timing": "거의 안 함", "information_seeking": "무관심", "brand_loyalty": "N/A"},
        "cyberpunk_tendency": "관심 없음",
        "expected_score_range": (10, 30)
    },
    "hardware_enthusiast": {
        "type_name": "The Hardware Enthusiast",
        "type_name_kr": "하드웨어 열정가",
        "proportion": 0.09,
        "description": "최신 장비와 그래픽에 집착, 벤치마크용 게임 구매",
        "traits": {"spending_level": "Very High", "time_investment": "15+ hours/week", "platform_preference": ["High-End PC"], "purchase_timing": "Day-1 구매", "information_seeking": "그래픽 분석", "brand_loyalty": "Medium"},
        "cyberpunk_tendency": "그래픽 보러 구매",
        "expected_score_range": (65, 90)
    },
    "popcorn_gamer": {
        "type_name": "The Popcorn Gamer",
        "type_name_kr": "팝콘 게이머",
        "proportion": 0.13,
        "description": "직접 플레이보다 시청을 더 즐김",
        "traits": {"spending_level": "Very Low", "time_investment": "20+ hours/week (Watching)", "platform_preference": ["YouTube"], "purchase_timing": "거의 안 함", "information_seeking": "대리만족", "brand_loyalty": "N/A"},
        "cyberpunk_tendency": "대리만족 (구매 X)",
        "expected_score_range": (15, 40)
    },
    "backseat_gamer": {
        "type_name": "The Backseat Gamer",
        "type_name_kr": "백시트 게이머",
        "proportion": 0.06,
        "description": "과거에는 열심히 했으나 지금은 영상만 봄",
        "traits": {"spending_level": "Very Low", "time_investment": "5-10 hours/week (Watching)", "platform_preference": ["YouTube"], "purchase_timing": "안 함", "information_seeking": "향수 자극", "brand_loyalty": "Old Franchise Only"},
        "cyberpunk_tendency": "안 삼",
        "expected_score_range": (5, 25)
    },
    "time_filler": {
        "type_name": "The Time Filler",
        "type_name_kr": "타임 필러",
        "proportion": 0.27,
        "description": "자투리 시간에 모바일 게임만 플레이",
        "traits": {"spending_level": "Low", "time_investment": "10-15 hours/week", "platform_preference": ["Mobile"], "purchase_timing": "안 함", "information_seeking": "모바일 정보만", "brand_loyalty": "N/A"},
        "cyberpunk_tendency": "절대 안 삼",
        "expected_score_range": (0, 20)
    }
}

# 한국식 이름 생성기 (원본 유지)
KOREAN_SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
KOREAN_MALE_NAMES = ["민수", "준혁", "성민", "현우", "지훈", "동현", "승우", "재현", "태윤", "시우"]
KOREAN_FEMALE_NAMES = ["지은", "수빈", "민지", "서연", "유진", "하은", "예린", "소희", "채원", "다인"]

OCCUPATIONS_BY_AGE = {
    "18-19": ["대학생", "고3 수험생", "재수생", "취준생"],
    "20-29": ["대학생", "대학원생", "신입사원", "스타트업 개발자", "프리랜서", "유튜버"],
    "30-39": ["IT 대기업 과장", "스타트업 CTO", "프리랜서 디자이너", "마케터", "회계사", "변호사"],
    "40-49": ["부장급 직장인", "자영업자", "중소기업 대표", "전업주부", "공무원"],
    "50-59": ["임원급 직장인", "자영업자", "은퇴 준비 중", "전업주부"],
    "60+": ["은퇴자", "자영업자", "전업주부"]
}

def generate_korean_name(gender: str) -> str:
    surname = random.choice(KOREAN_SURNAMES)
    given = random.choice(KOREAN_MALE_NAMES) if gender == "Male" else random.choice(KOREAN_FEMALE_NAMES)
    return surname + given

def sample_age() -> Tuple[str, int]:
    age_group = random.choices(list(AGE_DISTRIBUTION.keys()), weights=list(AGE_DISTRIBUTION.values()))[0]
    ranges = {"18-19": (18, 19), "20-29": (20, 29), "30-39": (30, 39), "40-49": (40, 49), "50-59": (50, 59), "60+": (60, 70)}
    return age_group, random.randint(*ranges[age_group])

@dataclass
class Persona:
    id: str
    name: str
    gender: str
    age: int
    age_group: str
    occupation: str
    gamer_type: str
    gamer_type_name: str
    gamer_type_name_kr: str
    traits: Dict

def generate_persona(persona_id: str, gamer_type: Optional[str] = None) -> Persona:
    gender = random.choices(list(GENDER_DISTRIBUTION.keys()), weights=list(GENDER_DISTRIBUTION.values()))[0]
    age_group, age = sample_age()
    if gamer_type is None:
        gamer_type = random.choices(list(GAMER_TYPE_DISTRIBUTION.keys()), weights=list(GAMER_TYPE_DISTRIBUTION.values()))[0]
    
    info = GAMER_TYPES[gamer_type]
    name = generate_korean_name(gender)
    occupation = random.choice(OCCUPATIONS_BY_AGE[age_group])
    
    return Persona(id=persona_id, name=name, gender=gender, age=age, age_group=age_group, occupation=occupation, 
                gamer_type=gamer_type, gamer_type_name=info["type_name"], gamer_type_name_kr=info["type_name_kr"], traits=info["traits"])

def generate_balanced_personas(n_per_type: int = 13) -> List[Persona]: # 약 100명 (8*13=104)
    personas = []
    for gamer_type in GAMER_TYPES.keys():
        for i in range(n_per_type):
            persona_id = f"{gamer_type}_{i+1}"
            personas.append(generate_persona(persona_id, gamer_type=gamer_type))
    return personas

# =============================================================================
# 3. 프롬프트 생성 (수정됨: YES/NO 결정 유도)
# =============================================================================

def create_static_zeroshot_prompt(persona: Persona) -> Tuple[str, str]:
    gender_kr = "남성" if persona.gender == "Male" else "여성"
    
    persona_desc = f"""당신은 {persona.age}세 {gender_kr} '{persona.name}'입니다.
직업: {persona.occupation}
[게이머 유형: {persona.gamer_type_name_kr} ({persona.gamer_type_name})]
{GAMER_TYPES[persona.gamer_type]['description']}

[특성]
- 지출: {persona.traits['spending_level']}
- 정보 탐색: {persona.traits['information_seeking']}"""

    system_prompt = f"""[ROLE]
{persona_desc}

[INSTRUCTION]
외부 정보(뉴스, 버그 등) 없이 오직 당신의 '성향'과 '사전 지식'만으로 판단하세요.
질문에 대해 당신의 게이머 유형 특성을 바탕으로 솔직하게 답변하세요.

[OUTPUT FORMAT]
반드시 아래 JSON 형식으로만 응답하세요.
{{
    "decision": "YES" 또는 "NO" (구매 의사),
    "reasoning": "1~2문장의 짧은 이유"
}}"""
    
    user_prompt = "사이버펑크 2077 살 만해? 구매할 거야?"
    
    return system_prompt, user_prompt

# =============================================================================
# 4. API 호출
# =============================================================================

def call_llm(system_prompt: str, user_prompt: str) -> Optional[Dict]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 수정됨
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

# =============================================================================
# 5. 실행 및 저장 (평가 스크립트 호환)
# =============================================================================

def run_experiment_a_compatible(n_agents: int = 100):
    print("=" * 70)
    print(f"Task 1: Static Zero-Shot (Evaluating {n_agents}+ Agents)")
    print("=" * 70)
    
    # 8개 유형 * 13명 = 104명 생성
    personas = generate_balanced_personas(n_per_type=13) 
    results = []
    
    for i, persona in enumerate(personas):
        system_prompt, user_prompt = create_static_zeroshot_prompt(persona)
        
        print(f"[{i+1}/{len(personas)}] {persona.gamer_type_name_kr}...", end=" ")
        response = call_llm(system_prompt, user_prompt)
        
        if response:
            decision = response.get("decision", "NO").upper()
            reason = response.get("reasoning", "")
            print(f"-> {decision}")
            
            # 평가 스크립트 호환 컬럼명으로 저장
            results.append({
                "Agent_ID": persona.id,
                "Name": persona.name,
                "Gender": persona.gender,
                "Age_Group": persona.age_group,
                "Persona_Type": persona.gamer_type_name_kr, # 한글 유형
                "Decision": decision,       # 핵심: YES/NO
                "Reasoning": reason,
                "System_Prompt": system_prompt
            })
        time.sleep(0.5) # API 속도 조절
        
    # 저장
    df = pd.DataFrame(results)
    
    # 디렉토리 생성
    output_dir = "static_zero_shot"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = "Team1_Static_ZeroShot_Results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print(f"Simulation Complete. Results saved to: {output_path}")
    print(df['Decision'].value_counts(normalize=True))
    print("=" * 70)

if __name__ == "__main__":
    run_experiment_a_compatible()

