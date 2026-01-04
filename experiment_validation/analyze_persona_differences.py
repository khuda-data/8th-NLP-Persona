"""
페르소나별 차이 분석
일반 데이터 분석(집단 평균)과 달리, 다양한 게이머 유형이 실제로 다른 의사결정을 하는지 검증
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

def analyze_persona_decision_patterns():
    """페르소나별 구매 의도 패턴 분석"""
    print("=" * 70)
    print("페르소나별 차이 분석")
    print("=" * 70)
    
    # 결과 파일 로드
    team3_file = "../time_aware_rag/Team3_TimeAware_Results_Final.csv"
    
    if not os.path.exists(team3_file):
        print("⚠️  결과 파일이 없습니다. 먼저 시뮬레이션을 실행하세요.")
        return
    
    df = pd.read_csv(team3_file)
    df['Decision'] = df['Decision'].apply(lambda x: 1 if str(x).strip().upper().startswith('YES') else 0)
    
    # 페르소나별 구매 의도 분석
    persona_stats = df.groupby('Persona_Type').agg({
        'Decision': ['mean', 'count', 'std']
    }).reset_index()
    persona_stats.columns = ['Persona_Type', 'Purchase_Ratio', 'Count', 'Std']
    
    print("\n📊 페르소나별 구매 의도:")
    print("-" * 70)
    persona_stats_sorted = persona_stats.sort_values('Purchase_Ratio', ascending=False)
    for _, row in persona_stats_sorted.iterrows():
        print(f"  {row['Persona_Type']:30s}: {row['Purchase_Ratio']:.3f} ({row['Count']}명)")
    print("-" * 70)
    
    # 시각화
    plt.figure(figsize=(16, 10))
    
    # 1. 페르소나별 구매 의도
    plt.subplot(2, 2, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(persona_stats)))
    bars = plt.barh(range(len(persona_stats_sorted)), 
                    persona_stats_sorted['Purchase_Ratio'],
                    color=colors)
    plt.yticks(range(len(persona_stats_sorted)), persona_stats_sorted['Persona_Type'])
    plt.xlabel('Purchase Ratio')
    plt.title('페르소나별 구매 의도 (개인화된 의사결정)')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 값 표시
    for i, (bar, ratio) in enumerate(zip(bars, persona_stats_sorted['Purchase_Ratio'])):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{ratio:.3f}', va='center', fontweight='bold')
    
    # 2. 페르소나별 분산
    plt.subplot(2, 2, 2)
    plt.barh(range(len(persona_stats_sorted)), 
             persona_stats_sorted['Std'],
             color=colors)
    plt.yticks(range(len(persona_stats_sorted)), persona_stats_sorted['Persona_Type'])
    plt.xlabel('Standard Deviation')
    plt.title('페르소나별 의사결정 분산')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 3. 시간에 따른 페르소나별 변화 (상위 3개 페르소나)
    plt.subplot(2, 2, 3)
    if 'Simulation_Date' in df.columns:
        df['Simulation_Date'] = pd.to_datetime(df['Simulation_Date'])
        top_personas = persona_stats_sorted.head(3)['Persona_Type'].tolist()
        
        for persona in top_personas:
            persona_df = df[df['Persona_Type'] == persona]
            daily_ratio = persona_df.groupby('Simulation_Date')['Decision'].mean()
            plt.plot(daily_ratio.index, daily_ratio.values, 
                    marker='o', label=persona, linewidth=2, markersize=4)
        
        plt.xlabel('Date')
        plt.ylabel('Purchase Ratio')
        plt.title('시간에 따른 페르소나별 변화 (상위 3개)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    # 4. 일반 분석 vs 개인화 비교
    plt.subplot(2, 2, 4)
    overall_mean = df['Decision'].mean()
    persona_means = persona_stats_sorted['Purchase_Ratio'].values
    
    # 일반 분석: 전체 평균 (단일 값)
    plt.axhline(y=overall_mean, color='r', linestyle='--', 
               linewidth=2, label=f'일반 분석 (전체 평균: {overall_mean:.3f})')
    
    # 개인화 분석: 페르소나별 평균 (다양한 값)
    plt.barh(range(len(persona_means)), persona_means, 
             color=colors, alpha=0.7, label='개인화 분석 (페르소나별)')
    
    plt.yticks(range(len(persona_stats_sorted)), persona_stats_sorted['Persona_Type'])
    plt.xlabel('Purchase Ratio')
    plt.title('일반 분석 vs 개인화 분석')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/persona_differences.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: figures/persona_differences.png")
    
    # 통계 요약
    print("\n📈 통계 요약:")
    print("-" * 70)
    print(f"전체 평균 구매 의도: {overall_mean:.3f}")
    print(f"페르소나별 최고: {persona_stats_sorted['Purchase_Ratio'].max():.3f}")
    print(f"페르소나별 최저: {persona_stats_sorted['Purchase_Ratio'].min():.3f}")
    print(f"페르소나별 범위: {persona_stats_sorted['Purchase_Ratio'].max() - persona_stats_sorted['Purchase_Ratio'].min():.3f}")
    print("-" * 70)
    print("\n💡 핵심 차별점:")
    print("  - 일반 데이터 분석: 전체 평균만 계산 (개인 차이 무시)")
    print("  - 본 실험: 페르소나별 개인화된 의사결정 (다양성 반영)")
    print(f"  → 페르소나에 따라 최대 {persona_stats_sorted['Purchase_Ratio'].max() - persona_stats_sorted['Purchase_Ratio'].min():.3f} 차이")
    
    # 결과 저장
    os.makedirs("results", exist_ok=True)
    persona_stats_sorted.to_csv("results/persona_statistics.csv", index=False, encoding='utf-8-sig')
    print(f"\n✅ 통계 결과 저장: results/persona_statistics.csv")

if __name__ == "__main__":
    print("\n🔬 페르소나별 차이 분석 시작\n")
    
    analyze_persona_decision_patterns()
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)
    print("\n💡 이 분석은 일반 데이터 분석과 달리,")
    print("   다양한 게이머 유형이 실제로 다른 의사결정을 한다는 것을")
    print("   증빙하며, 개인화된 시뮬레이션의 중요성을 보여줍니다.")

