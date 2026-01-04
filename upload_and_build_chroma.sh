#!/bin/bash
# 서버에 데이터 업로드 및 ChromaDB 구축 스크립트

SERVER="guswns0429@aurora.khu.ac.kr"
PORT="30080"
REMOTE_DIR="/data/guswns0429/8th-NLP-Persona"
LOCAL_CSV="datasets/Cyberpunk_2077_Steam_Reviews.csv"

echo "=========================================="
echo "🚀 서버에 데이터 업로드 및 ChromaDB 구축"
echo "=========================================="

# 1. 서버에 디렉토리 생성
echo "📁 서버 디렉토리 생성 중..."
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR/datasets"

# 2. CSV 파일 업로드 (압축해서 빠르게)
echo "📤 CSV 파일 업로드 중... (이 작업은 시간이 걸릴 수 있습니다)"
gzip -c $LOCAL_CSV | ssh -p $PORT $SERVER "gunzip > $REMOTE_DIR/datasets/Cyberpunk_2077_Steam_Reviews.csv"

# 3. 필요한 파일들 업로드
echo "📤 스크립트 파일 업로드 중..."
scp -P $PORT static_rag/build_chroma_db.py $SERVER:$REMOTE_DIR/static_rag/
scp -P $PORT requirements.txt $SERVER:$REMOTE_DIR/ 2>/dev/null || echo "⚠️  requirements.txt 없음 (스킵)"

# 4. 서버에서 ChromaDB 구축 실행
echo "🔨 서버에서 ChromaDB 구축 시작..."
ssh -p $PORT $SERVER << 'ENDSSH'
cd /data/guswns0429/8th-NLP-Persona
source .venv/bin/activate 2>/dev/null || python3 -m venv .venv && source .venv/bin/activate
pip install -q chromadb pandas sentence-transformers numpy
python static_rag/build_chroma_db.py
ENDSSH

echo ""
echo "✅ 완료! 서버에서 ChromaDB 구축이 완료되었습니다."
echo ""
echo "📥 ChromaDB 다운로드하려면:"
echo "   scp -r -P $PORT $SERVER:$REMOTE_DIR/datasets/chroma_db ./datasets/"

