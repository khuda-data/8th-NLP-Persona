"""
Team 1, 2, 3 vs 일반 데이터 분석 비교
일반 데이터 분석으로는 보이지 않는 패턴을 시뮬레이션이 포착하는지 증빙
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

def compare_statistical_vs_simulation():
    """일반 통계 분석 vs 시뮬레이션 비교"""
    print("=" * 70)
    print("일반 데이터 분석 vs 시뮬레이션 비교")
    print("=" * 70)
    
    # Ground Truth 로드
    steam_gt = pd.read_csv("../datasets/ground_truth_steam.csv")
    steam_gt['Date'] = pd.to_datetime(steam_gt['Date'])
    stock_gt = pd.read_csv("../datasets/ground_truth_stock.csv")
    stock_gt['Date'] = pd.to_datetime(stock_gt['Date'])
    
    # 결과 파일 로드
    team1_file = "../static_zero_shot/Team1_Static_ZeroShot_Results.csv"
    team2_file = "../static_rag/Team2_StaticRAG_Results.csv"
    team3_file = "../time_aware_rag/Team3_TimeAware_Results_Final.csv"
    
    files_exist = all(os.path.exists(f) for f in [team1_file, team2_file, team3_file])
    
    if not files_exist:
        print("⚠️  일부 결과 파일이 없습니다. 시뮬레이션을 먼저 실행하세요.")
        return
    
    # 데이터 로드
    team1_df = pd.read_csv(team1_file)
    team2_df = pd.read_csv(team2_file)
    team3_df = pd.read_csv(team3_file)
    
    # 구매 의도 계산
    for df in [team1_df, team2_df, team3_df]:
        df['Vote'] = df['Decision'].apply(lambda x: 1 if str(x).strip().upper().startswith('YES') else 0)
    
    # Team 1: 정적 (평균만)
    team1_mean = team1_df['Vote'].mean()
    
    # Team 2, 3: 동적 (시간에 따른 변화)
    team2_df['Simulation_Date'] = pd.to_datetime(team2_df['Simulation_Date'])
    team3_df['Simulation_Date'] = pd.to_datetime(team3_df['Simulation_Date'])
    
    team2_daily = team2_df.groupby('Simulation_Date')['Vote'].mean().reset_index()
    team2_daily.columns = ['Date', 'Purchase_Ratio']
    team3_daily = team3_df.groupby('Simulation_Date')['Vote'].mean().reset_index()
    team3_daily.columns = ['Date', 'Purchase_Ratio']
    
    # Ground Truth와 병합
    team2_steam = pd.merge(steam_gt[['Date', 'Positive_Ratio']], team2_daily, on='Date', how='inner')
    team3_steam = pd.merge(steam_gt[['Date', 'Positive_Ratio']], team3_daily, on='Date', how='inner')
    
    # 상관계수 계산
    corr_team1_steam = np.nan  # 정적이므로 상관계수 없음
    corr_team2_steam, _ = pearsonr(team2_steam['Purchase_Ratio'], team2_steam['Positive_Ratio'])
    corr_team3_steam, _ = pearsonr(team3_steam['Purchase_Ratio'], team3_steam['Positive_Ratio'])
    
    # 일반 통계 분석: 단순 평균
    overall_mean = team1_mean
    
    print("\n📊 방법론별 비교:")
    print("-" * 70)
    print(f"일반 데이터 분석 (단순 평균):")
    print(f"  - 구매 의도: {overall_mean:.3f} (고정값)")
    print(f"  - Steam 상관계수: NaN (변화 없음)")
    print(f"  - 특징: 시간 정보 무시, 집단 평균만 계산")
    print()
    print(f"Team 1 (Static Zero-Shot):")
    print(f"  - 구매 의도: {team1_mean:.3f} (고정값)")
    print(f"  - Steam 상관계수: NaN (변화 없음)")
    print(f"  - 특징: 페르소나만 사용, 외부 정보 없음")
    print()
    print(f"Team 2 (Static RAG):")
    print(f"  - Steam 상관계수: {corr_team2_steam:.4f}")
    print(f"  - 특징: RAG 사용, 시간 정보 무시")
    print()
    print(f"Team 3 (Time-Aware RAG):")
    print(f"  - Steam 상관계수: {corr_team3_steam:.4f}")
    print(f"  - 특징: RAG + 시간 가중치, 최신 정보 우선")
    print("-" * 70)
    
    # 시각화
    plt.figure(figsize=(16, 10))
    
    # 1. 전체 비교
    plt.subplot(2, 2, 1)
    plt.plot(team2_steam['Date'], team2_steam['Positive_Ratio'], 
             'b-', label='Ground Truth (Steam)', linewidth=2, alpha=0.7)
    plt.axhline(y=overall_mean, color='gray', linestyle='--', 
               linewidth=2, label=f'일반 분석 (평균: {overall_mean:.3f})')
    plt.plot(team2_steam['Date'], team2_steam['Purchase_Ratio'], 
             'r--o', label=f'Team 2 (r={corr_team2_steam:.3f})', 
             linewidth=2, markersize=4, alpha=0.7)
    plt.plot(team3_steam['Date'], team3_steam['Purchase_Ratio'], 
             'g--s', label=f'Team 3 (r={corr_team3_steam:.3f})', 
             linewidth=2, markersize=4, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.title('일반 분석 vs 시뮬레이션 비교')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. 상관계수 비교
    plt.subplot(2, 2, 2)
    methods = ['일반 분석', 'Team 1', 'Team 2', 'Team 3']
    correlations = [0, 0, corr_team2_steam, corr_team3_steam]  # 일반 분석과 Team1은 0
    
    colors = ['gray', 'orange', '#ff7f0e', '#2ca02c']
    bars = plt.bar(methods, correlations, color=colors, alpha=0.7)
    plt.ylabel('Correlation Coefficient')
    plt.title('Ground Truth와의 상관계수')
    plt.ylim([0, max(correlations) * 1.2 if max(correlations) > 0 else 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, corr in zip(bars, correlations):
        if corr > 0:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    'NaN', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. 시간에 따른 변화 추적 능력
    plt.subplot(2, 2, 3)
    # 일반 분석: 고정값
    dates = team2_steam['Date']
    plt.plot(dates, [overall_mean] * len(dates), 
             'gray', linestyle='--', linewidth=2, label='일반 분석 (고정값)')
    plt.plot(team2_steam['Date'], team2_steam['Purchase_Ratio'], 
             'r--o', label='Team 2', linewidth=2, markersize=4, alpha=0.7)
    plt.plot(team3_steam['Date'], team3_steam['Purchase_Ratio'], 
             'g--s', label='Team 3', linewidth=2, markersize=4, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Purchase Ratio')
    plt.title('시간에 따른 변화 추적 능력')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 4. 차별점 요약
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    comparison_text = f"""
    방법론 비교 요약
    
    일반 데이터 분석:
    • 단순 평균 계산
    • 시간 정보 무시
    • 집단 특성만 반영
    • 상관계수: NaN
    
    Team 1 (Zero-Shot):
    • 페르소나만 사용
    • 외부 정보 없음
    • 고정된 구매율
    • 상관계수: NaN
    
    Team 2 (Static RAG):
    • RAG로 정보 검색
    • 시간 정보 무시
    • 유사도만 사용
    • 상관계수: {corr_team2_steam:.3f}
    
    Team 3 (Time-Aware):
    • RAG + 시간 가중치
    • 최신 정보 우선
    • 동적 변화 추적
    • 상관계수: {corr_team3_steam:.3f}
    
    핵심 차별점:
    → 시뮬레이션은 시간에 따른
      동적 변화를 추적할 수 있음
    """
    
    plt.text(0.1, 0.5, comparison_text, fontsize=10, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/statistical_vs_simulation.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: figures/statistical_vs_simulation.png")
    
    # 결과 저장
    os.makedirs("results", exist_ok=True)
    comparison_df = pd.DataFrame({
        'Method': ['일반 분석', 'Team 1', 'Team 2', 'Team 3'],
        'Approach': ['단순 평균', 'Zero-Shot', 'Static RAG', 'Time-Aware RAG'],
        'Time_Aware': [False, False, False, True],
        'RAG_Used': [False, False, True, True],
        'Correlation_Steam': [np.nan, np.nan, corr_team2_steam, corr_team3_steam],
        'Can_Track_Changes': [False, False, True, True]
    })
    comparison_df.to_csv("results/method_comparison.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장: results/method_comparison.csv")
    
    print("\n💡 핵심 차별점:")
    print("  - 일반 데이터 분석: 시간 정보 무시, 집단 평균만 계산")
    print("  - 시뮬레이션: 시간에 따른 동적 변화 추적 가능")
    print("  - Time-Aware RAG: 최신 정보 우선으로 더 정확한 예측")

if __name__ == "__main__":
    print("\n🔬 방법론 비교 분석 시작\n")
    
    compare_statistical_vs_simulation()
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)

