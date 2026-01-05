"""
Team 2 vs Team 3 비교 분석
Time decay가 실제로 성능 개선을 가져오는지 검증
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

def compare_correlation_scores():
    """Team 2와 Team 3의 상관계수 비교"""
    print("=" * 70)
    print("Team 2 vs Team 3 성능 비교")
    print("=" * 70)
    
    # Ground Truth 로드
    steam_gt = pd.read_csv("../datasets/ground_truth_steam.csv")
    steam_gt['Date'] = pd.to_datetime(steam_gt['Date'])
    stock_gt = pd.read_csv("../datasets/ground_truth_stock.csv")
    stock_gt['Date'] = pd.to_datetime(stock_gt['Date'])
    
    # 결과 파일 로드
    team2_file = "../static_rag/Team2_StaticRAG_Results.csv"
    team3_file = "../time_aware_rag/Team3_TimeAware_Results_Final.csv"
    
    if not os.path.exists(team2_file) or not os.path.exists(team3_file):
        print("⚠️  결과 파일이 없습니다. 먼저 시뮬레이션을 실행하세요.")
        return
    
    team2_df = pd.read_csv(team2_file)
    team3_df = pd.read_csv(team3_file)
    
    # 구매 의도 계산
    for df in [team2_df, team3_df]:
        df['Vote'] = df['Decision'].apply(lambda x: 1 if str(x).strip().upper().startswith('YES') else 0)
        df['Simulation_Date'] = pd.to_datetime(df['Simulation_Date'])
    
    # 일별 구매 의도
    team2_daily = team2_df.groupby('Simulation_Date')['Vote'].mean().reset_index()
    team2_daily.columns = ['Date', 'Purchase_Ratio']
    team3_daily = team3_df.groupby('Simulation_Date')['Vote'].mean().reset_index()
    team3_daily.columns = ['Date', 'Purchase_Ratio']
    
    # Ground Truth와 병합
    team2_steam = pd.merge(steam_gt[['Date', 'Positive_Ratio']], team2_daily, on='Date', how='inner')
    team2_stock = pd.merge(stock_gt[['Date', 'Stock_Price']], team2_daily, on='Date', how='inner')
    team3_steam = pd.merge(steam_gt[['Date', 'Positive_Ratio']], team3_daily, on='Date', how='inner')
    team3_stock = pd.merge(stock_gt[['Date', 'Stock_Price']], team3_daily, on='Date', how='inner')
    
    # 상관계수 계산
    corr_team2_steam, _ = pearsonr(team2_steam['Purchase_Ratio'], team2_steam['Positive_Ratio'])
    corr_team2_stock, _ = pearsonr(team2_stock['Purchase_Ratio'], team2_stock['Stock_Price'])
    corr_team3_steam, _ = pearsonr(team3_steam['Purchase_Ratio'], team3_steam['Positive_Ratio'])
    corr_team3_stock, _ = pearsonr(team3_stock['Purchase_Ratio'], team3_stock['Stock_Price'])
    
    print("\n📊 상관계수 비교:")
    print("-" * 70)
    print(f"Team 2 (Static RAG):")
    print(f"  - Steam: {corr_team2_steam:.4f}")
    print(f"  - Stock: {corr_team2_stock:.4f}")
    print(f"\nTeam 3 (Time-Aware RAG):")
    print(f"  - Steam: {corr_team3_steam:.4f}")
    print(f"  - Stock: {corr_team3_stock:.4f}")
    print("-" * 70)
    
    improvement_steam = corr_team3_steam - corr_team2_steam
    improvement_stock = corr_team3_stock - corr_team2_stock
    
    print(f"\n📈 개선도:")
    print(f"  - Steam: {improvement_steam:+.4f} ({improvement_steam/corr_team2_steam*100:+.1f}%)")
    print(f"  - Stock: {improvement_stock:+.4f} ({improvement_stock/corr_team2_stock*100:+.1f}%)")
    
    # 시각화
    plt.figure(figsize=(16, 10))
    
    # 1. Steam 비교
    plt.subplot(2, 2, 1)
    plt.plot(team2_steam['Date'], team2_steam['Positive_Ratio'], 
             'b-', label='Ground Truth (Steam)', alpha=0.6, linewidth=2)
    plt.plot(team2_steam['Date'], team2_steam['Purchase_Ratio'], 
             'r--o', label=f'Team 2 (r={corr_team2_steam:.3f})', 
             linewidth=2, markersize=4, alpha=0.7)
    plt.plot(team3_steam['Date'], team3_steam['Purchase_Ratio'], 
             'g--s', label=f'Team 3 (r={corr_team3_steam:.3f})', 
             linewidth=2, markersize=4, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.title('Steam Positive Ratio 예측 비교')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. Stock 비교
    plt.subplot(2, 2, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(team2_stock['Date'], team2_stock['Stock_Price'], 
             'b-', label='Ground Truth (Stock)', alpha=0.6, linewidth=2)
    ax2.plot(team2_stock['Date'], team2_stock['Purchase_Ratio'], 
             'r--o', label=f'Team 2 (r={corr_team2_stock:.3f})', 
             linewidth=2, markersize=4, alpha=0.7)
    ax2.plot(team3_stock['Date'], team3_stock['Purchase_Ratio'], 
             'g--s', label=f'Team 3 (r={corr_team3_stock:.3f})', 
             linewidth=2, markersize=4, alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock Price', color='b')
    ax2.set_ylabel('Purchase Ratio', color='g')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='g')
    plt.title('Stock Price 예측 비교')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 3. 상관계수 비교
    plt.subplot(2, 2, 3)
    metrics = ['Steam', 'Stock']
    team2_scores = [corr_team2_steam, corr_team2_stock]
    team3_scores = [corr_team3_steam, corr_team3_stock]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, team2_scores, width, label='Team 2 (Static)', color='#ff7f0e', alpha=0.7)
    bars2 = plt.bar(x + width/2, team3_scores, width, label='Team 3 (Time-Aware)', color='#2ca02c', alpha=0.7)
    
    plt.ylabel('Correlation Coefficient')
    plt.title('상관계수 비교')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 개선도
    plt.subplot(2, 2, 4)
    improvements = [improvement_steam, improvement_stock]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = plt.bar(metrics, improvements, color=colors, alpha=0.7)
    plt.ylabel('Improvement (Δr)')
    plt.title('Time-Aware RAG의 개선도')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.4f}', ha='center', 
                va='bottom' if imp > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/team2_vs_team3_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: figures/team2_vs_team3_comparison.png")
    
    # 결과 저장
    os.makedirs("results", exist_ok=True)
    comparison_df = pd.DataFrame({
        'Metric': ['Steam', 'Stock'],
        'Team2_Correlation': [corr_team2_steam, corr_team2_stock],
        'Team3_Correlation': [corr_team3_steam, corr_team3_stock],
        'Improvement': [improvement_steam, improvement_stock],
        'Improvement_Percent': [improvement_steam/corr_team2_steam*100 if corr_team2_steam != 0 else 0,
                               improvement_stock/corr_team2_stock*100 if corr_team2_stock != 0 else 0]
    })
    comparison_df.to_csv("results/team2_vs_team3_comparison.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장: results/team2_vs_team3_comparison.csv")
    
    print("\n💡 핵심 차별점:")
    print("  - Team 2 (Static RAG): 시간 정보 무시, similarity만 사용")
    print("  - Team 3 (Time-Aware RAG): 시간 가중치 적용, 최신 정보 우선")
    if improvement_steam > 0 or improvement_stock > 0:
        print("  → Time decay가 실제로 성능 개선을 가져옴")

if __name__ == "__main__":
    print("\n🔬 Team 2 vs Team 3 비교 분석 시작\n")
    
    compare_correlation_scores()
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)

