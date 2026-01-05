"""
공통 LLM 설정 모듈
모든 팀(Team 1, 2, 3)이 동일한 LLM을 사용하도록 보장
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# =============================================================================
# LLM 설정 (모든 팀 공통)
# =============================================================================

# 기본값: OpenAI API 사용 (실험 공정성을 위해 통일)
USE_OLLAMA = False  # True로 변경하면 로컬 Ollama 사용

# Ollama 설정 (로컬 LLM 사용 시)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3:4b"

# OpenAI 설정 (기본값)
OPENAI_MODEL = "gpt-4o-mini"  # 모든 팀이 동일한 모델 사용

# Temperature 설정 (모든 팀 공통)
TEMPERATURE = 0.5

# =============================================================================
# LLM 클라이언트 초기화
# =============================================================================

def get_llm_client():
    """
    LLM 클라이언트 반환 (모든 팀 공통)
    
    Returns:
        OpenAI: OpenAI 클라이언트 객체
        str: 사용 중인 모델 이름
    """
    if USE_OLLAMA:
        print(f"🔹 Using Local LLM (Ollama): {OLLAMA_MODEL}")
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # Ollama는 api_key가 필요 없지만 클라이언트 호환성을 위해 더미 값
        )
        model_name = OLLAMA_MODEL
    else:
        print(f"🔸 Using OpenAI API: {OPENAI_MODEL}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env file. "
                "Please create a .env file with: OPENAI_API_KEY=your_key_here"
            )
        client = OpenAI(api_key=api_key)
        model_name = OPENAI_MODEL
    
    return client, model_name

# 전역 클라이언트 (선택적 사용)
_client, _model_name = None, None

def get_client():
    """전역 클라이언트 반환 (초기화 필요 시 자동 초기화)"""
    global _client, _model_name
    if _client is None:
        _client, _model_name = get_llm_client()
    return _client

def get_model_name():
    """현재 사용 중인 모델 이름 반환"""
    global _model_name
    if _model_name is None:
        _, _model_name = get_llm_client()
    return _model_name

