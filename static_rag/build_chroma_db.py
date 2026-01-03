import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import os
import argparse
import sys
import torch

# 설정 (Configuration)
CSV_PATH = os.path.join("datasets", "reviews", "Cyberpunk_2077_Steam_Reviews.csv")
DB_PATH = os.path.join("datasets", "chroma_db")
COLLECTION_NAME = "cyberpunk2077_reviews"
# Local Model Path (Relative to project root or absolute)
# User specified: models/Qwen3-Embedding-8B
MODEL_PATH = os.path.join("models", "Qwen3-Embedding-8B")
MODEL_NAME = "Qwen3-Embedding-8B" # For metadata or fallback

def process_reviews(csv_path):
    """
    CSV 파일을 읽고 처리합니다.
    유효한 리뷰를 필터링하고 날짜 형식을 변환합니다.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
        
    print(f"Total rows: {len(df)}")
    
    # 새로운 데이터셋 컬럼 매핑 확인
    # 'Review' 컬럼 존재 여부 확인
    if 'Review' not in df.columns:
        print(f"Error: 'Review' column not found. Columns: {df.columns.tolist()}")
        sys.exit(1)

    # 빈 리뷰 제거
    df = df.dropna(subset=['Review'])
    print(f"Rows after dropping empty reviews: {len(df)}")
    
    # 언어 필터링 등을 추가할 수 있으나, 현재 데이터셋은 대부분 영문으로 가정하고 진행
    
    return df

def parse_date_to_int(date_str):
    """
    날짜 문자열을 YYYYMMDD 정수형으로 변환합니다.
    'M/D/YYYY' (예: 12/9/2020) 등 다양한 형식을 지원합니다.
    """
    if pd.isna(date_str):
        return None
        
    formats = [
        '%m/%d/%Y',      # 12/9/2020
        '%Y-%m-%d',      # 2020-12-09
        '%d-%m-%Y',      # 09-12-2020
        '%B %d, %Y'      # December 9, 2020
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str(date_str).strip(), fmt)
            return int(dt.strftime('%Y%m%d'))
        except ValueError:
            continue
            
    return None

class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_path):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence_transformers: pip install sentence-transformers")
        
        print(f"Loading embedding model from: {model_path}")
        # trust_remote_code=True might be needed for some Qwen models
        self.model = SentenceTransformer(model_path, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, input: list) -> list:
        # Generate embeddings
        embeddings = self.model.encode(input, convert_to_tensor=False).tolist()
        return embeddings

def build_chroma_db(test_mode=False):
    # ChromaDB 클라이언트 초기화
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 임베딩 함수 설정 (Local Qwen Model)
    import torch # Ensure torch is imported for device check
    
    if os.path.exists(MODEL_PATH):
        print(f"Found local model at {MODEL_PATH}, using CustomEmbeddingFunction.")
        ef = CustomEmbeddingFunction(model_path=MODEL_PATH)
    else:
        print(f"Warning: Local model not found at {MODEL_PATH}. Falling back to default {MODEL_NAME}.")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 스키마 초기화를 위해 항상 기존 컬렉션 삭제
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Collection deletion skipped: {e}")

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    df = process_reviews(CSV_PATH)
    
    if test_mode:
        print("Test mode: Processing only first 5000 records.")
        df = df.head(5000)
    
    batch_size = 512
    total_docs = len(df)
    
    documents = []
    metadatas = []
    ids = []
    
    print("Starting ingestion...")
    
    for i, (idx, row) in enumerate(df.iterrows()):
        review_text = row['Review']
        date_val = row.get('Date Posted')
        
        # 날짜 파싱
        date_int = parse_date_to_int(date_val)
        if not date_int:
            # 날짜 파싱 실패 시 건너뜀
            continue
            
        # 메타데이터 구성
        # Rating: 'Recommended' -> True, 'Not Recommended' -> False
        rating_str = str(row.get('Rating', '')).lower()
        is_positive = 'recommended' in rating_str and 'not' not in rating_str
        
        # 플레이 타임 (Playtime)
        playtime = 0.0
        try:
            if 'Playtime' in row:
                # "10.5 hours" 문자열 등 처리
                pt_str = str(row['Playtime']).replace('hours', '').strip()
                playtime = float(pt_str)
        except:
            pass
            
        metadata = {
            "date": date_int,
            "rating": rating_str,
            "voted_up": is_positive,
            "playtime": playtime,
            "source": "steam_new_dataset"
        }
        
        # ID: ReviewID가 있으면 사용, 없으면 인덱스 사용
        doc_id = str(row['ReviewID']) if 'ReviewID' in row else f"rev_{i}"
        
        documents.append(review_text)
        metadatas.append(metadata)
        ids.append(doc_id)
        
        if len(documents) >= batch_size:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            documents = []
            metadatas = []
            ids = []
            print(f"Processed {i + 1}/{total_docs} reviews...", end='\r')
            
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    print(f"\nIngestion complete. Total documents in collection: {collection.count()}")

    if test_mode:
        verify_insertion(collection)

def verify_insertion(collection):
    print("\n--- Verification ---")
    print("Querying for 'cyberpunk'...")
    results = collection.query(
        query_texts=["Great open world game"],
        n_results=1
    )
    print("Result:")
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build ChromaDB from new reviews CSV.')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    
    build_chroma_db(test_mode=args.test)
