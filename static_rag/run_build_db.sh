#!/bin/bash
#SBATCH -J build_chroma_db
#SBATCH --gres=gpu:1
#SBATCH -p batch_ugrad                
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/build_db_%A.out
#SBATCH -e logs/build_db_%A.err

# 작업 정보 출력
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# 로그 디렉토리 생성
mkdir -p logs

# Conda 환경 활성화 (base 환경 사용 or 프로젝트 환경)
# run_training.sh 참조: source /data/mcladinz/anaconda3/etc/profile.d/conda.sh
source /data/mcladinz/anaconda3/etc/profile.d/conda.sh
conda activate base 

# GPU 확인
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo ""

# 모델 경로 확인 (models/Qwen3-Embedding-8B)
MODEL_DIR="models/Qwen3-Embedding-8B"
echo "Checking model directory: $MODEL_DIR"
if [ -d "$MODEL_DIR" ]; then
    echo "✓ Custom Model found at $MODEL_DIR"
else
    echo "✗ Model directory not found at $MODEL_DIR"
    echo "  Please ensure the model is placed correctly."
    exit 1
fi
echo ""

# DB 빌드 실행
echo "=========================================="
echo "Starting ChromaDB Build..."
echo "=========================================="

# PYTHONPATH 설정 (프로젝트 루트)
export PYTHONPATH=$PYTHONPATH:.

python static_rag/build_chroma_db.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "ChromaDB Build completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ChromaDB Build failed!"
    echo "=========================================="
    exit 1
fi

# 최종 결과 요약
echo ""
echo "Job Summary"
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo "Logs: logs/build_db_${SLURM_JOB_ID}.out"
