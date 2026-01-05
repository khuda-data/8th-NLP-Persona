#!/usr/bin/env python3
"""
GPT-4o-mini 비용 계산 스크립트
"""

import pandas as pd

# GPT-4o-mini 가격 (2024년 기준)
# Input: $0.15 per 1M tokens
# Output: $0.60 per 1M tokens
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60

def estimate_tokens(text):
    """텍스트의 대략적인 토큰 수 추정 (1 token ≈ 4 characters)"""
    return len(text) / 4

def calculate_team1_cost():
    """Team 1 비용 계산"""
    print("="*70)
    print("Team 1: Static Zero-Shot")
    print("="*70)
    
    # Team 1 프롬프트 예시
    system_prompt = """[ROLE]
You are a 33 year old Male named 'Michael Davis'.
Occupation: Accountant
[Gamer Type: The Ultimate Gamer]
A passionate gamer who spares no time or money on games.

[Traits]
- Spending Level: Very High
- Information Seeking: Buys regardless of reviews

[INSTRUCTION]
Make a decision based SOLELY on your 'traits' and 'prior knowledge' without any external information (news, bugs, reviews, etc.).
Answer honestly based on your gamer persona.

[OUTPUT FORMAT]
You MUST respond in the following JSON format:
{
    "decision": "YES" or "NO" (Purchase Intention),
    "reasoning": "A short reason (1-2 sentences)"
}"""
    
    user_prompt = "Is 'Cyberpunk 2077' worth buying? Will you buy it?"
    
    # 응답 예시
    response = '{"decision": "YES", "reasoning": "As an ultimate gamer, I see the potential for an immersive experience in Cyberpunk 2077 and I am willing to invest in it regardless of past issues or reviews."}'
    
    input_tokens = estimate_tokens(system_prompt + user_prompt)
    output_tokens = estimate_tokens(response)
    
    # Team 1: 104 agents × 1 call
    num_calls = 104
    
    total_input_tokens = input_tokens * num_calls
    total_output_tokens = output_tokens * num_calls
    
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost
    
    print(f"에이전트 수: {num_calls}명")
    print(f"API 호출 수: {num_calls}회")
    print(f"평균 입력 토큰: {input_tokens:.0f} tokens/call")
    print(f"평균 출력 토큰: {output_tokens:.0f} tokens/call")
    print(f"총 입력 토큰: {total_input_tokens:,.0f} tokens")
    print(f"총 출력 토큰: {total_output_tokens:,.0f} tokens")
    print(f"입력 비용: ${input_cost:.4f}")
    print(f"출력 비용: ${output_cost:.4f}")
    print(f"총 비용: ${total_cost:.4f}")
    print()
    
    return total_cost, num_calls

def calculate_team2_cost():
    """Team 2 비용 계산"""
    print("="*70)
    print("Team 2: Static RAG")
    print("="*70)
    
    # Team 2 프롬프트 예시 (리뷰 컨텍스트 포함)
    system_prompt = """[ROLE]
You are a 33 year old Male.
Personality: 'The Ultimate Gamer' (A passionate gamer who spares no time or money on games.)

[DATE]
Today is 2020-12-10.

[SEARCH RESULTS]
Reviews selected based on your interests and recentness (Time-Weighted):
- [2020-12-09] Great open world game with amazing graphics...
- [2020-12-08] The game has some bugs but the story is engaging...
- [2020-12-07] Performance issues on low-end hardware...
- [2020-12-06] Worth buying if you like RPGs...
- [2020-12-05] Not recommended due to optimization problems...

[TASK]
Decide to buy 'Cyberpunk 2077' or not based strictly on the reviews above.
- The reviews are filtered by relevance and recency.
- Trust these reviews as the most important information available to you.

[OUTPUT]
JSON only:
{
    "decision": "YES" or "NO",
    "reasoning": "Explain why based on the reviews."
}"""
    
    user_prompt = ""  # System prompt에 모든 내용 포함
    
    # 응답 예시
    response = '{"decision": "YES", "reasoning": "The reviews indicate that Cyberpunk 2077 has significantly improved since its initial release, with players praising its graphics, world design, and overall attention to detail."}'
    
    input_tokens = estimate_tokens(system_prompt + user_prompt)
    output_tokens = estimate_tokens(response)
    
    # Team 2: 62 dates × 104 agents
    num_dates = 62
    num_agents = 104
    num_calls = num_dates * num_agents
    
    total_input_tokens = input_tokens * num_calls
    total_output_tokens = output_tokens * num_calls
    
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost
    
    print(f"시뮬레이션 날짜: {num_dates}일")
    print(f"에이전트 수: {num_agents}명")
    print(f"API 호출 수: {num_calls:,}회")
    print(f"평균 입력 토큰: {input_tokens:.0f} tokens/call")
    print(f"평균 출력 토큰: {output_tokens:.0f} tokens/call")
    print(f"총 입력 토큰: {total_input_tokens:,.0f} tokens")
    print(f"총 출력 토큰: {total_output_tokens:,.0f} tokens")
    print(f"입력 비용: ${input_cost:.4f}")
    print(f"출력 비용: ${output_cost:.4f}")
    print(f"총 비용: ${total_cost:.4f}")
    print()
    
    return total_cost, num_calls

def calculate_team3_cost():
    """Team 3 비용 계산 (Team 2와 동일한 구조)"""
    print("="*70)
    print("Team 3: Time-Aware RAG")
    print("="*70)
    
    # Team 3 프롬프트는 Team 2와 유사 (리뷰 선택 방식만 다름)
    system_prompt = """[ROLE]
You are a 33 year old Male.
Personality: 'The Ultimate Gamer' (A passionate gamer who spares no time or money on games.)

[DATE]
Today is 2020-12-10.

[SEARCH RESULTS]
Reviews selected based on your interests and recentness (Time-Weighted):
- [2020-12-09] Great open world game with amazing graphics...
- [2020-12-08] The game has some bugs but the story is engaging...
- [2020-12-07] Performance issues on low-end hardware...
- [2020-12-06] Worth buying if you like RPGs...
- [2020-12-05] Not recommended due to optimization problems...

[TASK]
Decide to buy 'Cyberpunk 2077' or not based strictly on the reviews above.
- The reviews are filtered by relevance and recency (Time-Decay applied).
- Trust these reviews as the most important information available to you.

[OUTPUT]
JSON only:
{
    "decision": "YES" or "NO",
    "reasoning": "Explain why based on the reviews."
}"""
    
    user_prompt = ""
    response = '{"decision": "YES", "reasoning": "The reviews indicate that Cyberpunk 2077 has significantly improved since its initial release, with players praising its graphics, world design, and overall attention to detail."}'
    
    input_tokens = estimate_tokens(system_prompt + user_prompt)
    output_tokens = estimate_tokens(response)
    
    # Team 3: 62 dates × 104 agents
    num_dates = 62
    num_agents = 104
    num_calls = num_dates * num_agents
    
    total_input_tokens = input_tokens * num_calls
    total_output_tokens = output_tokens * num_calls
    
    input_cost = (total_input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost
    
    print(f"시뮬레이션 날짜: {num_dates}일")
    print(f"에이전트 수: {num_agents}명")
    print(f"API 호출 수: {num_calls:,}회")
    print(f"평균 입력 토큰: {input_tokens:.0f} tokens/call")
    print(f"평균 출력 토큰: {output_tokens:.0f} tokens/call")
    print(f"총 입력 토큰: {total_input_tokens:,.0f} tokens")
    print(f"총 출력 토큰: {total_output_tokens:,.0f} tokens")
    print(f"입력 비용: ${input_cost:.4f}")
    print(f"출력 비용: ${output_cost:.4f}")
    print(f"총 비용: ${total_cost:.4f}")
    print()
    
    return total_cost, num_calls

def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("  💰 GPT-4o-mini 실험 비용 계산")
    print("="*70)
    print("\n가격 정보 (2024년 기준):")
    print(f"  - Input: ${INPUT_COST_PER_1M} per 1M tokens")
    print(f"  - Output: ${OUTPUT_COST_PER_1M} per 1M tokens")
    print()
    
    cost1, calls1 = calculate_team1_cost()
    cost2, calls2 = calculate_team2_cost()
    cost3, calls3 = calculate_team3_cost()
    
    total_cost = cost1 + cost2 + cost3
    total_calls = calls1 + calls2 + calls3
    
    print("="*70)
    print("  📊 전체 요약")
    print("="*70)
    print(f"Team 1: ${cost1:.4f} ({calls1:,} calls)")
    print(f"Team 2: ${cost2:.4f} ({calls2:,} calls)")
    print(f"Team 3: ${cost3:.4f} ({calls3:,} calls)")
    print("-" * 70)
    print(f"총 비용: ${total_cost:.4f}")
    print(f"총 API 호출: {total_calls:,}회")
    print("="*70)
    print()
    print("⚠️  참고:")
    print("  - 실제 비용은 프롬프트 길이와 응답 길이에 따라 달라질 수 있습니다.")
    print("  - 리뷰 컨텍스트 길이는 검색 결과에 따라 변동됩니다.")
    print("  - 위 계산은 평균적인 추정치입니다.")

if __name__ == "__main__":
    main()

