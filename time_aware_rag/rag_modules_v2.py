import chromadb
import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch

from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY

CHROMA_DB_PATH = "/local_datasets/a2024105535/chroma_db"
COLLECTION_NAME = "cyberpunk2077_reviews"

def get_chroma_client():
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {CHROMA_DB_PATH}")
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_embedding_function():
    # Bi-Encoder (1차 검색용)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

class RAGRetriever:
    def __init__(self):
        print("⏳ [Init] DB 로딩 중...")
        self.client = get_chroma_client()
        self.embedding_fn = get_embedding_function()
        
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            print(f"✅ ChromaDB 로드 완료. Docs: {self.collection.count()}")
        except Exception as e:
            raise ValueError(f"컬렉션 로드 실패: {e}")

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"⏳ [Init] Cross-Encoder({model_name}) 로딩 중... (GPU 사용)")
        self.reranker = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Cross-Encoder 로드 완료!")

    def _sigmoid(self, x):
        """Cross-Encoder 점수(-10~10)를 확률(0~1)로 변환"""
        return 1 / (1 + np.exp(-x))

    def retrieve_reviews(self, agent, current_date_str: str, top_k_final: int = 5, decay_rate: float = 0.01):
        """
        [로직 순서]
        1. 1차 검색 (Bi-Encoder): ChromaDB에서 100개 가져옴
        2. 2차 리랭킹 (Cross-Encoder): AI가 정밀 채점
        3. 시간 가중치: 채점 결과에 시간 점수 반영
        """
        
        # 1. 날짜 처리
        try:
            curr_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
            curr_date_int = int(curr_dt.strftime("%Y%m%d"))
        except:
            return []

        # 2. 쿼리 선정 (속도를 위해 2~3개만 사용 권장)
        if hasattr(agent, "search_queries"):
            queries = agent.search_queries
        else:
            queries = [GENERAL_QUERY]

        selected_queries = random.sample(queries, min(3, len(queries)))
        if GENERAL_QUERY not in selected_queries:
            selected_queries.append(GENERAL_QUERY)

        # ---------------------------------------------------------
        # [Step 1] Broad Retrieval (ChromaDB)
        # ---------------------------------------------------------
        results = self.collection.query(
            query_texts=selected_queries,
            n_results=300,
            include=["documents", "metadatas"],
            where={"date": {"$lte": curr_date_int}} 
        )

        # 후보군 중복 제거 및 통합
        unique_candidates = {}
        if results['documents']:
            for i in range(len(results['documents'])): 
                docs = results['documents'][i]
                metas = results['metadatas'][i]
                for doc, meta in zip(docs, metas):
                    if doc not in unique_candidates:
                        unique_candidates[doc] = meta

        if not unique_candidates:
            return []

        candidates_list = list(unique_candidates.items()) # [(doc, meta), ...]
        
        # ---------------------------------------------------------
        # [Step 2] Cross-Encoder Scoring
        # ---------------------------------------------------------
        
        # (질문, 리뷰) 쌍 만들기
        reference_query = GENERAL_QUERY 
        pairs = [[reference_query, doc] for doc, meta in candidates_list]
        
        # AI 모델 예측
        ce_scores = self.reranker.predict(pairs) 

        # ---------------------------------------------------------
        # [Step 3] Time-Decay Application
        # ---------------------------------------------------------
        final_results = []
        
        for i, (doc, meta) in enumerate(candidates_list):
            # A. Cross-Encoder 점수를 0~1 확률로 변환
            relevance_prob = self._sigmoid(ce_scores[i])
            
            # B. 시간 차이 계산
            review_date_int = meta.get('date')
            if review_date_int:
                try:
                    review_dt = datetime.strptime(str(review_date_int), "%Y%m%d")
                    days_diff = (curr_dt - review_dt).days
                except:
                    days_diff = 0
            else:
                days_diff = 0
            if days_diff < 0: days_diff = 0

            # C. 시간 가중치 적용
            time_weight = np.exp(-decay_rate * days_diff)
            
            # D. 최종 점수 (관련성 × 최신성)
            final_score = relevance_prob * time_weight
            
            final_results.append({
                "review": doc,
                "date_int": review_date_int,
                "final_score": final_score
            })

        # 정렬 및 반환
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        top_docs = final_results[:top_k_final]

        formatted_results = []
        for item in top_docs:
            d_str = str(item['date_int'])
            date_formatted = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
            formatted_results.append(f"- [{date_formatted}] {item['review'][:400]}...")
            
        return formatted_results

if __name__ == "__main__":
    retriever = RAGRetriever()
    
    # 가짜 에이전트로 테스트
    class MockAgent:
        search_queries = ["optimization", "bugs", "gameplay"]
    
    # 테스트 날짜 (출시 초기 vs 최근)
    test_date = "2023-12-01"
    print(f"\n🔍 Testing retrieval for {test_date}...")
    
    reviews = retriever.retrieve_reviews(MockAgent(), test_date, top_k_final=3)
    
    for r in reviews:
        print(r)

