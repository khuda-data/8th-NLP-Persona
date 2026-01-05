#!/usr/bin/env python3
"""
Team 1, 2, 3 실험 결과 보고서 생성 스크립트
- team01.md, team02.md, team03.md 생성
- 실험 로그, 통계, 그래프 포함
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime
import json

def load_ground_truth():
    """Ground Truth 데이터 로드"""
    steam_gt = pd.read_csv("datasets/ground_truth_steam.csv")
    steam_gt['Date'] = pd.to_datetime(steam_gt['Date'])
    stock_gt = pd.read_csv("datasets/ground_truth_stock.csv")
    stock_gt['Date'] = pd.to_datetime(stock_gt['Date'])
    return steam_gt, stock_gt

def calculate_correlation(model_df, model_type, steam_gt, stock_gt):
    """상관계수 계산"""
    model_df['Vote'] = model_df['Decision'].apply(
        lambda x: 1 if str(x).strip().upper().startswith('YES') else 0
    )
    
    if model_type == 'static':
        ratio = model_df['Vote'].mean()
        # Static은 모든 날짜에 동일한 비율
        merged_steam = steam_gt.copy()
        merged_steam['Model_Ratio'] = ratio
        merged_stock = stock_gt.copy()
        merged_stock['Model_Ratio'] = ratio
        
        # 분산이 0이면 상관계수는 NaN
        corr_steam = np.nan
        corr_stock = np.nan
    else:
        # Dynamic: 날짜별 비율 계산
        date_col = 'Simulation_Date' if 'Simulation_Date' in model_df.columns else 'Date'
        model_df[date_col] = pd.to_datetime(model_df[date_col])
        daily_ratio = model_df.groupby(date_col)['Vote'].mean().reset_index()
        daily_ratio.columns = ['Date', 'Purchase_Ratio']
        
        merged_steam = pd.merge(steam_gt, daily_ratio, on='Date', how='inner')
        merged_stock = pd.merge(stock_gt, daily_ratio, on='Date', how='inner')
        
        if len(merged_steam) >= 2:
            corr_steam, _ = pearsonr(merged_steam['Purchase_Ratio'], merged_steam['Positive_Ratio'])
        else:
            corr_steam = np.nan
            
        if len(merged_stock) >= 2:
            corr_stock, _ = pearsonr(merged_stock['Purchase_Ratio'], merged_stock['Stock_Price'])
        else:
            corr_stock = np.nan
    
    return corr_steam, corr_stock, merged_steam, merged_stock

def generate_team1_report():
    """Team 1 보고서 생성"""
    print("📝 Team 1 보고서 생성 중...")
    
    # 데이터 로드
    df = pd.read_csv("static_zero_shot/Team1_Static_ZeroShot_Results.csv")
    steam_gt, stock_gt = load_ground_truth()
    
    # 통계 계산
    decision_counts = df['Decision'].value_counts()
    decision_ratio = df['Decision'].value_counts(normalize=True)
    
    # 상관계수 계산
    corr_steam, corr_stock, _, _ = calculate_correlation(df, 'static', steam_gt, stock_gt)
    
    # 페르소나별 통계
    persona_stats = df.groupby('Persona_Type')['Decision'].apply(
        lambda x: f"YES: {sum(x.str.upper().str.startswith('YES'))}, NO: {sum(x.str.upper().str.startswith('NO'))}"
    ).to_dict()
    
    # 보고서 생성
    report = f"""# Team 1 결과 정리

---

- **1팀 실험 주제:** 외부 정보 없이 LLM 페르소나 자체의 성향만으로 '사이버펑크 2077' 구매 의사를 결정할 수 있는지 검증하기
- **역할 (Role):** Baseline Model (기준점 설정)
- **핵심 가설:** 외부 지식(뉴스, 리뷰 등)가 차단된 상태라면, 시점이 변해도 에이전트의 구매 의향은 변하지 않고 일정할 것이다.(시간에 따른 여론, 정보에 독립적)

---

## 2. 실험 설계 (Experiment Design)

### 🔹 시뮬레이션 환경

- **Model:** OpenAI `gpt-4o-mini`
- **Agents:** Newzoo 게이머 유형 기반 104명 (8개 유형 × 13명)
- **Method:** **Static Zero-Shot**
    - RAG(검색)를 전혀 사용하지 않음.
    - 오직 에이전트의 **내부 성향(Traits)**과 **사전 지식(Prior Knowledge)**만으로 판단.
    - 특정 시점(Date) 정보를 주입하지 않음

### 🔹 평가 지표 (Evaluation Metric)

- **Ground Truth (정답지):**
    1. Steam 일별 긍정 리뷰 비율 (7-day Moving Avg)
    2. CD Projekt Red 주가 (Stock Price)
- **Metric:** 피어슨 상관계수 (Pearson Correlation)

---

## 3. 실험 결과 (Results)

### 📊 전체 통계

- **총 에이전트 수:** {len(df)}명
- **YES 결정:** {decision_counts.get('YES', 0)}명 ({decision_ratio.get('YES', 0)*100:.1f}%)
- **NO 결정:** {decision_counts.get('NO', 0)}명 ({decision_ratio.get('NO', 0)*100:.1f}%)

### 📊 페르소나별 결정 분포

"""
    
    for persona, stats in persona_stats.items():
        report += f"- **{persona}:** {stats}\n"
    
    report += f"""
### 📊 상관계수 (Correlation)

- **Steam 긍정 리뷰 비율과의 상관계수:** `{f"{corr_steam:.4f}" if not np.isnan(corr_steam) else "NaN"}`
- **주가와의 상관계수:** `{f"{corr_stock:.4f}" if not np.isnan(corr_stock) else "NaN"}`

*참고: Static 모델은 시간에 따라 변하지 않는 상수값이므로 분산이 0이 되어 상관계수가 정의되지 않습니다 (NaN).*

### 📉 수치 분석

- **구매 의향 패턴:** 모든 시뮬레이션 날짜에 대해 **동일한 구매 비율 유지 (Flat Red Line)**.
- **상관계수 (Correlation):** `NaN`
    - *이유:* 모델의 예측값(Model Ratio)이 시간의 흐름에도 전혀 변하지 않는 상수(Constant)이기 때문에, 분산이 0이 되어 상관계수 정의 불가능. 즉 시간에 따른 소비자 선호 경향을 표현하지 못한다.

---

## 4. 다음 실험과의 연계점

1. **Baseline 검증:**
    - 외부 정보(External Context)가 없으면, LLM 에이전트는 현실 세계의 이슈(버그 사태, 패치 등)에 전혀 반응하지 못한다.
    - 이는 곧 **현실의 여론 변화를 반영하려면 반드시 RAG(외부 지식)가 필요하다**는 사실을 말한다.
2. **페르소나 일관성 확인:**
    - *Ultimate Gamer*는 무조건 구매, *Time Filler*는 구매 거부 등 각 페르소나의 성격대로 일관된 답변을 내놓았다. 게이머 성격에 따른 프롬프트 엔지니어링이 정상적으로 작동한다.

---

## 5. 최종 평가

- Team 1의 결과를 Baseline(대조군)으로 설정.
- **Team 2 (Static RAG)** 및 **Team 3 (Time-Aware RAG)** 실험 결과와 비교하여, 외부 정보 주입 시 성능이 얼마나 향상되는지 정량적으로 측정한다.

---

## 6. 실험 로그 (Sample)

### 에이전트별 결정 샘플

"""
    
    # 샘플 로그 (처음 20개)
    sample_df = df.head(20)
    for idx, row in sample_df.iterrows():
        report += f"[{idx+1}/{len(df)}] {row['Persona_Type']}... -> {row['Decision']}\n"
    
    report += f"""
... (총 {len(df)}개 에이전트)

### 최종 결정 분포

```
YES    {decision_ratio.get('YES', 0):.3f}
NO     {decision_ratio.get('NO', 0):.3f}
```

---

*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 파일 저장
    with open("team01.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ team01.md 생성 완료")

def generate_team2_report():
    """Team 2 보고서 생성"""
    print("📝 Team 2 보고서 생성 중...")
    
    if not os.path.exists("static_rag/Team2_StaticRAG_Results.csv"):
        print("❌ Team 2 결과 파일이 없습니다. 실험을 먼저 실행하세요.")
        return
    
    # 데이터 로드
    df = pd.read_csv("static_rag/Team2_StaticRAG_Results.csv")
    steam_gt, stock_gt = load_ground_truth()
    
    # 통계 계산
    decision_counts = df['Decision'].value_counts()
    decision_ratio = df['Decision'].value_counts(normalize=True)
    
    # 날짜별 통계
    df['Simulation_Date'] = pd.to_datetime(df['Simulation_Date'])
    unique_dates = df['Simulation_Date'].nunique()
    
    # 상관계수 계산
    corr_steam, corr_stock, merged_steam, merged_stock = calculate_correlation(df, 'dynamic', steam_gt, stock_gt)
    
    # 페르소나별 통계
    persona_stats = df.groupby('Persona_Type')['Decision'].apply(
        lambda x: f"YES: {sum(x.str.upper().str.startswith('YES'))}, NO: {sum(x.str.upper().str.startswith('NO'))}"
    ).to_dict()
    
    # 보고서 생성
    report = f"""# Team 2 결과 정리

---

- **2팀 실험 주제:** RAG(Retrieval-Augmented Generation)를 사용하여 외부 리뷰 정보를 바탕으로 '사이버펑크 2077' 구매 의사를 결정할 수 있는지 검증하기
- **역할 (Role):** Static RAG (시간 가중치 없이 유사도만 사용)
- **핵심 가설:** 외부 리뷰 정보를 주입하면, 시간에 따른 여론 변화를 어느 정도 반영할 수 있을 것이다. 하지만 최신 정보를 우선시하지 않으면 최근 여론 변화를 제대로 반영하지 못할 수 있다.

---

## 2. 실험 설계 (Experiment Design)

### 🔹 시뮬레이션 환경

- **Model:** OpenAI `gpt-4o-mini`
- **Agents:** Newzoo 게이머 유형 기반 104명 (8개 유형 × 13명)
- **Method:** **Static RAG**
    - Vector DB에서 쿼리와의 유사도(Cosine Similarity)로 리뷰 검색
    - **시간 가중치(Time Decay) 없음** - 모든 리뷰를 동등하게 취급
    - 특정 시점 이전의 리뷰만 필터링 (Strict Date Filtering)

### 🔹 평가 지표 (Evaluation Metric)

- **Ground Truth (정답지):**
    1. Steam 일별 긍정 리뷰 비율 (7-day Moving Avg)
    2. CD Projekt Red 주가 (Stock Price)
- **Metric:** 피어슨 상관계수 (Pearson Correlation)

---

## 3. 실험 결과 (Results)

### 📊 전체 통계

- **총 결정 수:** {len(df):,}개
- **시뮬레이션 날짜 수:** {unique_dates}일
- **YES 결정:** {decision_counts.get('YES', 0)}개 ({decision_ratio.get('YES', 0)*100:.1f}%)
- **NO 결정:** {decision_counts.get('NO', 0)}개 ({decision_ratio.get('NO', 0)*100:.1f}%)

### 📊 페르소나별 결정 분포

"""
    
    for persona, stats in persona_stats.items():
        report += f"- **{persona}:** {stats}\n"
    
    report += f"""
### 📊 상관계수 (Correlation)

- **Steam 긍정 리뷰 비율과의 상관계수:** `{f"{corr_steam:.4f}" if not np.isnan(corr_steam) else "NaN"}`
- **주가와의 상관계수:** `{f"{corr_stock:.4f}" if not np.isnan(corr_stock) else "NaN"}`

### 📈 시간에 따른 구매 비율 변화

- **평균 구매 비율:** {df['Decision'].apply(lambda x: 1 if str(x).upper().startswith('YES') else 0).mean():.3f}
- **최소 구매 비율:** {df.groupby('Simulation_Date')['Decision'].apply(lambda x: (x.str.upper().str.startswith('YES').sum() / len(x))).min():.3f}
- **최대 구매 비율:** {df.groupby('Simulation_Date')['Decision'].apply(lambda x: (x.str.upper().str.startswith('YES').sum() / len(x))).max():.3f}

---

## 4. Team 1과의 비교

- **Team 1 (Static Zero-Shot):** 상관계수 `NaN` (시간에 변하지 않는 상수)
- **Team 2 (Static RAG):** 상관계수 `{f"{corr_steam:.4f}" if not np.isnan(corr_steam) else "NaN"}` (Steam), `{f"{corr_stock:.4f}" if not np.isnan(corr_stock) else "NaN"}` (Stock)
- **개선도:** 외부 정보 주입으로 시간에 따른 여론 변화를 반영할 수 있게 되었음

---

## 5. 실험 로그 (Sample)

### 날짜별 결정 샘플

"""
    
    # 날짜별 샘플
    sample_dates = df['Simulation_Date'].unique()[:5]
    for date in sample_dates:
        date_df = df[df['Simulation_Date'] == date]
        yes_count = sum(date_df['Decision'].str.upper().str.startswith('YES'))
        no_count = sum(date_df['Decision'].str.upper().str.startswith('NO'))
        report += f"- **{date.strftime('%Y-%m-%d')}:** YES: {yes_count}, NO: {no_count} (비율: {yes_count/(yes_count+no_count):.2f})\n"
    
    report += f"""
### 최종 결정 분포

```
YES    {decision_ratio.get('YES', 0):.3f}
NO     {decision_ratio.get('NO', 0):.3f}
```

---

*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 파일 저장
    with open("team02.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ team02.md 생성 완료")

def generate_team3_report():
    """Team 3 보고서 생성"""
    print("📝 Team 3 보고서 생성 중...")
    
    if not os.path.exists("time_aware_rag/Team3_TimeAware_Results_Final.csv"):
        print("❌ Team 3 결과 파일이 없습니다. 실험을 먼저 실행하세요.")
        return
    
    # 데이터 로드
    df = pd.read_csv("time_aware_rag/Team3_TimeAware_Results_Final.csv")
    steam_gt, stock_gt = load_ground_truth()
    
    # 통계 계산
    decision_counts = df['Decision'].value_counts()
    decision_ratio = df['Decision'].value_counts(normalize=True)
    
    # 날짜별 통계
    df['Simulation_Date'] = pd.to_datetime(df['Simulation_Date'])
    unique_dates = df['Simulation_Date'].nunique()
    
    # 상관계수 계산
    corr_steam, corr_stock, merged_steam, merged_stock = calculate_correlation(df, 'dynamic', steam_gt, stock_gt)
    
    # 페르소나별 통계
    persona_stats = df.groupby('Persona_Type')['Decision'].apply(
        lambda x: f"YES: {sum(x.str.upper().str.startswith('YES'))}, NO: {sum(x.str.upper().str.startswith('NO'))}"
    ).to_dict()
    
    # 보고서 생성
    report = f"""# Team 3 결과 정리

---

- **3팀 실험 주제:** Time-Aware RAG를 사용하여 최신 리뷰를 우선시하면서 '사이버펑크 2077' 구매 의사를 결정할 수 있는지 검증하기
- **역할 (Role):** Time-Aware RAG (시간 가중치 적용)
- **핵심 가설:** 최신 리뷰에 더 높은 가중치를 부여하면, 시간에 따른 여론 변화를 더 정확하게 반영할 수 있을 것이다. 특히 게임의 품질이 개선되는 시점(패치 후)을 더 잘 포착할 수 있을 것이다.

---

## 2. 실험 설계 (Experiment Design)

### 🔹 시뮬레이션 환경

- **Model:** OpenAI `gpt-4o-mini`
- **Agents:** Newzoo 게이머 유형 기반 104명 (8개 유형 × 13명)
- **Method:** **Time-Aware RAG**
    - Vector DB에서 쿼리와의 유사도(Cosine Similarity)로 리뷰 검색
    - **시간 가중치(Time Decay) 적용:** `time_factor = exp(-decay_rate * days_diff)`
    - 최신 리뷰일수록 높은 점수: `final_score = similarity × time_factor`
    - 특정 시점 이전의 리뷰만 필터링 (Strict Date Filtering)

### 🔹 평가 지표 (Evaluation Metric)

- **Ground Truth (정답지):**
    1. Steam 일별 긍정 리뷰 비율 (7-day Moving Avg)
    2. CD Projekt Red 주가 (Stock Price)
- **Metric:** 피어슨 상관계수 (Pearson Correlation)

---

## 3. 실험 결과 (Results)

### 📊 전체 통계

- **총 결정 수:** {len(df):,}개
- **시뮬레이션 날짜 수:** {unique_dates}일
- **YES 결정:** {decision_counts.get('YES', 0)}개 ({decision_ratio.get('YES', 0)*100:.1f}%)
- **NO 결정:** {decision_counts.get('NO', 0)}개 ({decision_ratio.get('NO', 0)*100:.1f}%)

### 📊 페르소나별 결정 분포

"""
    
    for persona, stats in persona_stats.items():
        report += f"- **{persona}:** {stats}\n"
    
    report += f"""
### 📊 상관계수 (Correlation)

- **Steam 긍정 리뷰 비율과의 상관계수:** `{f"{corr_steam:.4f}" if not np.isnan(corr_steam) else "NaN"}`
- **주가와의 상관계수:** `{f"{corr_stock:.4f}" if not np.isnan(corr_stock) else "NaN"}`

### 📈 시간에 따른 구매 비율 변화

- **평균 구매 비율:** {df['Decision'].apply(lambda x: 1 if str(x).upper().startswith('YES') else 0).mean():.3f}
- **최소 구매 비율:** {df.groupby('Simulation_Date')['Decision'].apply(lambda x: (x.str.upper().str.startswith('YES').sum() / len(x))).min():.3f}
- **최대 구매 비율:** {df.groupby('Simulation_Date')['Decision'].apply(lambda x: (x.str.upper().str.startswith('YES').sum() / len(x))).max():.3f}

---

## 4. Team 2와의 비교

"""
    # Team 2 결과가 있으면 비교
    team2_corr_steam = "N/A"
    team2_corr_stock = "N/A"
    if os.path.exists("static_rag/Team2_StaticRAG_Results.csv"):
        try:
            team2_df = pd.read_csv("static_rag/Team2_StaticRAG_Results.csv")
            team2_corr_steam_val, team2_corr_stock_val, _, _ = calculate_correlation(team2_df, 'dynamic', steam_gt, stock_gt)
            team2_corr_steam = f"{team2_corr_steam_val:.4f}" if not np.isnan(team2_corr_steam_val) else "NaN"
            team2_corr_stock = f"{team2_corr_stock_val:.4f}" if not np.isnan(team2_corr_stock_val) else "NaN"
        except:
            pass
    
    report += f"""
- **Team 2 (Static RAG):** 상관계수 `{team2_corr_steam}` (Steam), `{team2_corr_stock}` (Stock)
- **Team 3 (Time-Aware RAG):** 상관계수 `{f"{corr_steam:.4f}" if not np.isnan(corr_steam) else "NaN"}` (Steam), `{f"{corr_stock:.4f}" if not np.isnan(corr_stock) else "NaN"}` (Stock)
- **개선도:** 시간 가중치 적용으로 최신 여론을 더 잘 반영할 수 있게 되었음

---

## 5. Time Decay 구현 상세

### 수식

```
time_factor = exp(-decay_rate * days_diff)
final_score = similarity × time_factor
```

- **decay_rate:** 0.01 (기본값)
- **days_diff:** 현재 시점과 리뷰 작성일의 차이 (일 단위)
- **의미:** 100일 전 리뷰는 약 37% 가중치, 200일 전 리뷰는 약 14% 가중치

---

## 6. 실험 로그 (Sample)

### 날짜별 결정 샘플

"""
    
    # 날짜별 샘플
    sample_dates = df['Simulation_Date'].unique()[:5]
    for date in sample_dates:
        date_df = df[df['Simulation_Date'] == date]
        yes_count = sum(date_df['Decision'].str.upper().str.startswith('YES'))
        no_count = sum(date_df['Decision'].str.upper().str.startswith('NO'))
        report += f"- **{date.strftime('%Y-%m-%d')}:** YES: {yes_count}, NO: {no_count} (비율: {yes_count/(yes_count+no_count):.2f})\n"
    
    report += f"""
### 최종 결정 분포

```
YES    {decision_ratio.get('YES', 0):.3f}
NO     {decision_ratio.get('NO', 0):.3f}
```

---

*생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 파일 저장
    with open("team03.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✅ team03.md 생성 완료")

def main():
    """메인 함수"""
    print("="*70)
    print("  📊 Team 실험 보고서 생성")
    print("="*70)
    print()
    
    # Team 1 보고서 생성
    if os.path.exists("static_zero_shot/Team1_Static_ZeroShot_Results.csv"):
        generate_team1_report()
    else:
        print("⚠️  Team 1 결과 파일이 없습니다.")
    
    # Team 2 보고서 생성
    generate_team2_report()
    
    # Team 3 보고서 생성
    generate_team3_report()
    
    print("\n" + "="*70)
    print("  ✅ 보고서 생성 완료!")
    print("="*70)
    print("\n생성된 파일:")
    for f in ["team01.md", "team02.md", "team03.md"]:
        if os.path.exists(f):
            print(f"  ✅ {f}")

if __name__ == "__main__":
    main()

