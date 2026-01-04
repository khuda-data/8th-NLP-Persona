"""
AI 모델별 차이 분석 (선택 사항)
다른 LLM 모델을 사용했을 때 결과가 유의미하게 달라지는지 확인
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

def analyze_model_sensitivity():
    """
    AI 모델별 차이 분석
    참고: 현재는 모든 팀이 동일한 모델(gpt-4o-mini)을 사용하므로
    이 분석은 이론적 비교만 제공합니다.
    """
    print("=" * 70)
    print("AI 모델별 차이 분석 (이론적 비교)")
    print("=" * 70)
    
    print("\n📊 현재 설정:")
    print("-" * 70)
    print("모든 팀이 동일한 LLM 모델 사용: gpt-4o-mini")
    print("목적: 실험 공정성 보장 (대조군 생성)")
    print("-" * 70)
    
    print("\n💡 모델별 차이 가능성:")
    print("-" * 70)
    print("1. 모델 성능 차이:")
    print("   - 더 큰 모델(gpt-4o)은 더 정확한 판단 가능")
    print("   - 하지만 실험 목적은 'RAG 방식 차이' 검증이므로 모델 통일 필요")
    print()
    print("2. Temperature 차이:")
    print("   - 높은 temperature: 더 다양한 응답")
    print("   - 낮은 temperature: 더 일관된 응답")
    print("   - 현재: 모든 팀 0.5로 통일")
    print()
    print("3. 프롬프트 이해도:")
    print("   - 모델마다 프롬프트 해석 방식 다를 수 있음")
    print("   - 동일 모델 사용으로 이 변수 제거")
    print("-" * 70)
    
    print("\n🔬 실험 설계 관점:")
    print("-" * 70)
    print("현재 실험의 목적:")
    print("  → Time-Aware RAG의 효과성 검증")
    print("  → 모델 차이가 아닌 RAG 방식 차이만 반영해야 함")
    print()
    print("따라서:")
    print("  ✅ 모든 팀이 동일한 모델 사용 (공정성 보장)")
    print("  ✅ 모델 차이로 인한 편향 제거")
    print("  ✅ 성능 차이는 오직 RAG 방식 차이만 반영")
    print("-" * 70)
    
    print("\n📝 향후 연구 가능성:")
    print("-" * 70)
    print("만약 모델별 차이를 분석하고 싶다면:")
    print("  1. 동일한 RAG 방식으로 다른 모델 실험")
    print("  2. 모델 성능이 RAG 효과에 미치는 영향 분석")
    print("  3. 하지만 현재 실험 목적과는 별개")
    print("-" * 70)
    
    # 시각화: 이론적 비교
    plt.figure(figsize=(14, 8))
    
    # 모델별 가상 성능 비교 (이론적)
    plt.subplot(2, 2, 1)
    models = ['gpt-3.5-turbo', 'gpt-4o-mini\n(현재)', 'gpt-4o', 'Claude-3']
    hypothetical_corr = [0.45, 0.52, 0.58, 0.55]  # 가상의 값
    
    colors = ['lightblue', '#2ca02c', 'orange', 'purple']
    bars = plt.bar(models, hypothetical_corr, color=colors, alpha=0.7)
    bars[1].set_color('#2ca02c')  # 현재 모델 강조
    bars[1].set_edgecolor('black')
    bars[1].set_linewidth(2)
    
    plt.ylabel('Hypothetical Correlation')
    plt.title('모델별 가상 성능 비교 (이론적)')
    plt.ylim([0, 0.7])
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    # 값 표시
    for bar, corr in zip(bars, hypothetical_corr):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{corr:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 실험 설계 비교
    plt.subplot(2, 2, 2)
    plt.axis('off')
    
    design_text = """
    실험 설계 비교
    
    ❌ 잘못된 설계:
    • Team 2: gpt-3.5-turbo
    • Team 3: gpt-4o
    → 모델 차이와 RAG 차이를
      구분할 수 없음
    
    ✅ 올바른 설계 (현재):
    • Team 2: gpt-4o-mini
    • Team 3: gpt-4o-mini
    → 모델 차이 제거,
      RAG 차이만 반영
    
    결과:
    → 성능 차이는 오직
      Time-Aware RAG 효과
    """
    
    plt.text(0.1, 0.5, design_text, fontsize=11, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Temperature 영향
    plt.subplot(2, 2, 3)
    temps = [0.0, 0.3, 0.5, 0.7, 1.0]
    consistency = [1.0, 0.95, 0.85, 0.70, 0.50]  # 일관성 (가상)
    diversity = [0.0, 0.3, 0.5, 0.7, 1.0]  # 다양성 (가상)
    
    plt.plot(temps, consistency, 'b-o', label='일관성', linewidth=2, markersize=6)
    plt.plot(temps, diversity, 'r-s', label='다양성', linewidth=2, markersize=6)
    plt.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='현재 설정')
    plt.xlabel('Temperature')
    plt.ylabel('Score')
    plt.title('Temperature에 따른 응답 특성 (이론적)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 실험 공정성 체크리스트
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    checklist_text = """
    실험 공정성 체크리스트
    
    ✅ LLM 모델: 통일됨
    ✅ Temperature: 통일됨
    ✅ 페르소나: 통일됨
    ✅ 쿼리 생성: 통일됨
    ✅ 평가 기준: 통일됨
    
    차별점:
    → 오직 RAG 방식만 다름
      (Time decay 적용 여부)
    
    결론:
    → 실험 결과는 RAG 방식
      차이만 반영함
    """
    
    plt.text(0.1, 0.5, checklist_text, fontsize=11, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/model_differences_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: figures/model_differences_analysis.png")
    
    print("\n💡 결론:")
    print("  - 현재 실험은 모델 차이를 제거하여 RAG 방식 차이만 검증")
    print("  - 모델별 차이 분석은 별도의 연구 주제")
    print("  - 실험 공정성 보장을 위해 모델 통일이 필수적")

if __name__ == "__main__":
    print("\n🔬 AI 모델별 차이 분석 시작\n")
    
    analyze_model_sensitivity()
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)
    print("\n⚠️  참고: 이 분석은 이론적 비교입니다.")
    print("   실제로는 모든 팀이 동일한 모델을 사용하므로")
    print("   모델 차이로 인한 편향은 없습니다.")

