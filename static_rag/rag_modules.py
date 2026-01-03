import chromadb
import os
import openai
from chromadb.utils import embedding_functions
import sys

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY

# RAG 모듈 (RAG Modules)
# 이 모듈은 ChromaDB와의 연결 및 검색 로직을 담당합니다.
# 데이터베이스 경로와 컬렉션 이름을 설정하고, 특정 날짜 이전의 리뷰만 검색하도록 필터링합니다.

# ChromaDB 경로 설정 (Set ChromaDB Path)
CHROMA_DB_PATH = "datasets/chroma_db"
COLLECTION_NAME = "cyberpunk2077_reviews"

def get_chroma_client():
    """
    ChromaDB 영구 클라이언트(PersisentClient)를 반환합니다.
    Returns:
        chromadb.PersistentClient: ChromaDB 클라이언트 객체
    """
    # 데이터베이스 디렉토리가 존재하는지 확인
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {CHROMA_DB_PATH}")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def get_embedding_function():
    """
    임베딩 함수를 설정하여 반환합니다.
    기존 ChromaDB 컬렉션이 'all-MiniLM-L6-v2' (SentenceTransformer)로 생성되었으므로,
    이를 맞춰서 사용합니다.
    Returns:
        embedding_functions.SentenceTransformerEmbeddingFunction: 임베딩 함수 객체
    """
    # OpenAI 대신 로컬 모델 사용 (DB와의 호환성 유지)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

class RAGRetriever:
    def __init__(self):
        """
        RAGRetriever 초기화
        ChromaDB 클라이언트와 컬렉션을 로드합니다.
        """
        self.client = get_chroma_client()
        self.embedding_fn = get_embedding_function()
        
        # 컬렉션 가져오기 (Get Collection)
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            print(f"Collection '{COLLECTION_NAME}' loaded successfully.")
        except Exception as e:
            raise ValueError(f"컬렉션 '{COLLECTION_NAME}'을 로드하는 중 오류가 발생했습니다: {e}")

    def retrieve_reviews(self, query_text, current_date, top_k=5):
        """
        주어진 쿼리와 날짜를 기준으로 관련 리뷰를 검색합니다.
        
        Args:
            query_text (str): 검색할 쿼리 텍스트 (영어)
            current_date (str): 시뮬레이션 현재 날짜 (YYYY-MM-DD 형식). 이 날짜 이전의 리뷰만 검색됨.
            top_k (int): 반환할 상위 결과 개수
            
        Returns:
            list: 검색된 문서(리뷰) 리스트
        """
        # 날짜 포맷 변환: YYYY-MM-DD -> YYYYMMDD (int)
        # ChromaDB 메타데이터가 int형 날짜로 저장되어 있으므로 이에 맞춰 변환
        try:
            date_int = int(current_date.replace("-", ""))
        except ValueError:
            print(f"Warning: Invalid date format {current_date}. Using current timestamp.")
            date_int = 20250101 # Fallback
            
        # 날짜 필터링: 현재 날짜보다 작거나 같은(lte) 데이터만 검색
        where_filter = {"date": {"$lte": date_int}}
        
        # print(f"Retrieving for query: '{query_text}' with date filter <= {date_int}")
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter
        )
        
        if results['documents'] and results['documents'][0]:
            # Team 3 스타일: "- [Date] Review..." 형식으로 변환
            formatted_results = []
            
            # documents[0], metadatas[0]는 리스트 형태임
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            for doc, meta in zip(docs, metas):
                date_int = meta.get('date', 0)
                # 정수형 날짜(YYYYMMDD)를 다시 문자열(YYYY-MM-DD)로 변환
                date_str = str(date_int)
                if len(date_str) == 8:
                    date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                
                # 리뷰 길이를 400자로 제한 (Team 3와 동일)
                review_snippet = doc[:400]
                formatted_results.append(f"- [{date_str}] {review_snippet}...")
                
            return formatted_results
        else:
            return []

if __name__ == "__main__":
    # 테스트 코드 (Test Code)
    try:
        retriever = RAGRetriever()
        
        # 테스트 쿼리
        test_query = "Cyberpunk 2077 bugs and glitches"
        test_date = "2023-12-31"
        
        print(f"\n--- Testing Retrieval for date: {test_date} ---")
        reviews = retriever.retrieve_reviews(test_query, test_date, top_k=3)
        
        for i, review in enumerate(reviews):
            print(f"\nReview {i+1}:")
            print(review[:200] + "...") # 앞부분만 출력
            
    except Exception as e:
        print(f"Error during test: {e}")
