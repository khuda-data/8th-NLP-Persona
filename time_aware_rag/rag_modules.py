import chromadb
import os
import sys
import random
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY

# ChromaDB 경로 및 컬렉션 설정
CHROMA_DB_PATH = "datasets/chroma_db"
COLLECTION_NAME = "cyberpunk2077_reviews"

def get_chroma_client():
    """ChromaDB PersistentClient 반환"""
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def get_embedding_function():
    """SentenceTransformer 임베딩 함수 반환"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

class RAGRetriever:
    def __init__(self):
        """RAGRetriever 초기화 및 컬렉션 로드"""
        self.client = get_chroma_client()
        self.embedding_fn = get_embedding_function()
        
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            print(f"Collection '{COLLECTION_NAME}' loaded successfully.")
        except Exception as e:
            raise ValueError(f"컬렉션 '{COLLECTION_NAME}'을 로드하는 중 오류가 발생했습니다: {e}")

    # ===========================================================
    # 핵심 차별점: Time-Aware Weighted Score 적용
    # ===========================================================
    # Team 2 (static_rag)와의 차이:
    # - Team 2: 쿼리당 top_k개만 검색, similarity만 사용
    # - Team 3: 쿼리당 100개 검색 후 similarity × time_factor로 재랭킹
    # 
    # Time Decay 공식:
    #   time_factor = exp(-decay_rate * days_diff)
    #   final_score = similarity * time_factor
    #
    # decay_rate=0.01일 때:
    #   - 0일 전: time_factor = 1.00 (100%)
    #   - 70일 전: time_factor ≈ 0.50 (50%, half-life)
    #   - 100일 전: time_factor ≈ 0.37 (37%)
    # ===========================================================
    def retrieve_reviews(self, agent, current_date_str: str, top_k_final: int = 5, decay_rate: float = 0.01):
        """
        Time-Aware Weighted RAG 검색
        
        Team 3의 핵심 차별점: 시간 가중치를 적용하여 최근 리뷰에 높은 우선순위 부여
        
        Args:
            agent: Persona 객체 (search_queries 속성 포함)
            current_date_str: 시뮬레이션 현재 날짜 (YYYY-MM-DD 형식)
            top_k_final: 최종 반환할 리뷰 개수 (기본값: 5)
            decay_rate: 시간 감쇠율 (기본값: 0.01, half-life ≈ 70일)
        
        Returns:
            list: "- [Date] Review text..." 형식의 리뷰 문자열 리스트
        
        Process:
            1. 각 쿼리마다 n_results=100으로 넓은 후보 풀 검색
            2. 각 후보에 대해 similarity × time_factor 계산
            3. final_score 기준으로 정렬하여 상위 top_k_final개 선택
            4. 중복 제거 (동일 리뷰는 최고 점수만 유지)
        """
        current_ts = int(pd.to_datetime(current_date_str).timestamp())

        # 쿼리 선정 (Team 2와 동일: 4개 랜덤 + GENERAL_QUERY)
        # 공정성 보장: Team 2와 동일한 쿼리 선택 전략 사용
        selected_queries = random.sample(agent.search_queries, min(4, len(agent.search_queries)))
        selected_queries.append(GENERAL_QUERY)

        candidate_pool = []

        # 각 쿼리마다 넓은 후보 풀 검색 (Team 2는 top_k만, Team 3는 100개)
        for query in selected_queries:
            results = self.collection.query(
                query_texts=[query],
                n_results=100,  # 넓은 풀에서 검색 (Team 2와의 차이점)
                include=["documents", "metadatas", "distances"],
                where={"timestamp": {"$lte": current_ts}}  # 현재 날짜 이전 리뷰만
            )

            if results['documents'] and results['documents'][0]:
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                    # Cosine similarity 계산 (Team 2와 동일)
                    # ChromaDB의 distance는 cosine distance이므로 similarity = 1 - distance
                    similarity = max(0, 1 - dist)
                    
                    # 시간 차이 계산 (일 단위)
                    review_ts = meta['timestamp']
                    days_diff = max(0, (current_ts - review_ts) / (60 * 60 * 24))
                    
                    # Time decay factor 계산 (핵심 차별점)
                    # Exponential decay: 최근 리뷰일수록 높은 가중치
                    time_factor = np.exp(-decay_rate * days_diff)
                    
                    # 최종 점수: similarity × time_factor
                    # Team 2는 similarity만 사용, Team 3는 여기에 time_factor 곱함
                    final_score = similarity * time_factor

                    candidate_pool.append({
                        "review": doc,
                        "date_str": meta.get('date_str', 'Unknown'),
                        "final_score": final_score,
                        "similarity": similarity,
                        "days_diff": int(days_diff)
                    })

        if not candidate_pool:
            return []

        # 중복 제거 및 최고 점수 유지
        unique_candidates = {}
        for item in candidate_pool:
            text = item['review']
            if text not in unique_candidates or item['final_score'] > unique_candidates[text]['final_score']:
                unique_candidates[text] = item

        candidates_list = list(unique_candidates.values())
        candidates_list.sort(key=lambda x: x['final_score'], reverse=True)
        final_docs = candidates_list[:top_k_final]

        # Team3 스타일로 변환
        return [f"- [{item['date_str']}] {item['review'][:400]}..." for item in final_docs]

# ===========================================================
# 테스트용 main
# ===========================================================
if __name__ == "__main__":
    try:
        retriever = RAGRetriever()
        
        # 테스트용 Agent 객체 예시
        class Agent:
            def __init__(self):
                self.search_queries = ["graphics", "bugs", "storyline"]

        agent = Agent()
        test_date = "2023-12-31"
        reviews = retriever.retrieve_reviews(agent, current_date_str=test_date, top_k_final=3)
        
        print(f"\n--- Testing Retrieval for date: {test_date} ---")
        for i, review in enumerate(reviews):
            print(f"\nReview {i+1}:")
            print(review[:200] + "...")
            
    except Exception as e:
        print(f"Error during test: {e}")
