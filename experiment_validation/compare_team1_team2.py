"""
Team 1 vs Team 2 Comparison Analysis
Impact of RAG (Retrieved Information) on Purchase Intent
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure utils can be imported if needed (though we mainly use pandas here)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' # Generic safe font

def compare_team1_team2():
    print("=" * 70)
    print("Team 1 (Zero-Shot) vs Team 2 (Static RAG) Comparison")
    print("=" * 70)

    # 1. Load Data
    # Paths relative to this script location (experiment_validation/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    team1_file = os.path.join(project_root, "static_zero_shot", "Team1_Static_ZeroShot_Results.csv")
    team2_file = os.path.join(project_root, "static_rag", "Team2_StaticRAG_Results.csv")

    if not os.path.exists(team1_file):
        print(f"⚠️  Result file not found: {team1_file}")
        return
    if not os.path.exists(team2_file):
        print(f"⚠️  Result file not found: {team2_file}")
        return

    print(f"Loading Team 1 results from: {team1_file}")
    try:
        df1 = pd.read_csv(team1_file)
    except Exception as e:
        print(f"Error loading Team 1 file: {e}")
        return

    print(f"Loading Team 2 results from: {team2_file}")
    try:
        df2 = pd.read_csv(team2_file)
    except Exception as e:
        print(f"Error loading Team 2 file: {e}")
        return

    # 2. Preprocess
    # Convert 'Decision' to numeric (YES=1, NO=0)
    for df in [df1, df2]:
        df['Vote'] = df['Decision'].apply(
            lambda x: 1 if str(x).strip().upper().startswith('YES') else 0
        )
    
    # Team 2 has dates, convert to datetime
    if 'Simulation_Date' in df2.columns:
        df2['Simulation_Date'] = pd.to_datetime(df2['Simulation_Date'])

    # 3. Analyze by Persona Type
    # Team 1: Static Purchase Rate per Persona
    t1_persona = df1.groupby('Persona_Type')['Vote'].mean().reset_index()
    t1_persona.rename(columns={'Vote': 'T1_Rate'}, inplace=True)

    # Team 2: Average Purchase Rate per Persona (across all dates)
    t2_persona = df2.groupby('Persona_Type')['Vote'].mean().reset_index()
    t2_persona.rename(columns={'Vote': 'T2_Rate'}, inplace=True)

    # Merge
    persona_comp = pd.merge(t1_persona, t2_persona, on='Persona_Type', how='outer')
    persona_comp['Diff'] = persona_comp['T2_Rate'] - persona_comp['T1_Rate']
    
    print("\n📊 Impact of RAG by Persona Type:")
    print("-" * 60)
    print(f"{'Persona Type':<30} | {'Team 1':<8} | {'Team 2':<8} | {'Diff':<8}")
    print("-" * 60)
    for _, row in persona_comp.iterrows():
        print(f"{row['Persona_Type']:<30} | {row['T1_Rate']:.3f}    | {row['T2_Rate']:.3f}    | {row['Diff']:+.3f}")
    print("-" * 60)

    # 4. Overall Statistics
    t1_mean = df1['Vote'].mean()
    t2_mean = df2['Vote'].mean()
    
    print(f"\n📈 Overall Purchase Rate:")
    print(f"  - Team 1 (Baseline): {t1_mean:.3f}")
    print(f"  - Team 2 (With RAG): {t2_mean:.3f}")
    print(f"  - Change: {t2_mean - t1_mean:+.3f}")

    # 5. Visualizations
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(15, 6))

    # --- Plot 1: Bar Chart by Persona ---
    plt.subplot(1, 2, 1)
    
    # Prepare data for seaborn barplot
    comp_melt = persona_comp.melt(id_vars=['Persona_Type', 'Diff'], 
                                  value_vars=['T1_Rate', 'T2_Rate'], 
                                  var_name='Team', value_name='Rate')
    
    sns.barplot(data=comp_melt, x='Persona_Type', y='Rate', hue='Team', palette={'T1_Rate': 'gray', 'T2_Rate': '#1f77b4'})
    
    plt.title('Purchase Intent by Gamer Type (Static vs RAG)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Purchase Probability')
    plt.ylim(0, 1.05)
    plt.legend(title='Method', labels=['Team 1 (No Info)', 'Team 2 (RAG)'])
    plt.grid(axis='y', alpha=0.3)

    # --- Plot 2: Time Series Analysis ---
    plt.subplot(1, 2, 2)
    
    # Team 2 Daily Trend
    if 'Simulation_Date' in df2.columns:
        daily_trend = df2.groupby('Simulation_Date')['Vote'].mean().reset_index()
        
        plt.plot(daily_trend['Simulation_Date'], daily_trend['Vote'], 
                 marker='o', linestyle='-', color='#1f77b4', label='Team 2 (Static RAG)', alpha=0.8)
        
        # Team 1 Baseline Line
        plt.axhline(y=t1_mean, color='gray', linestyle='--', linewidth=2, label=f'Team 1 Baseline ({t1_mean:.2f})')
        
        plt.title(f'Temporal Stability Check (Team 2)\nvs Baseline (Team 1)')
        plt.xlabel('Simulation Date')
        plt.ylabel('Daily Purchase Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "No Date Info in Team 2 Results", ha='center')

    plt.tight_layout()
    plot_path = "figures/team1_vs_team2_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graph saved to: {plot_path}")

    # Save summary csv
    csv_path = "results/team1_vs_team2_summary.csv"
    persona_comp.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Summary saved to: {csv_path}")

if __name__ == "__main__":
    compare_team1_team2()
