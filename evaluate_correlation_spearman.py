
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 스타일 설정
sns.set_theme(style="whitegrid")

def load_ground_truth(path, value_col):
    """
    Ground Truth 파일을 로드하고 날짜 형식으로 변환합니다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"GT 파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    # 날짜순 정렬 (merge_asof를 위해 필수)
    return df[['Date', value_col]].sort_values('Date')

def calculate_daily_ratio(df_model):
    """
    모델 결과에서 일별 매수 비율(Purchase Ratio)을 계산합니다.
    """
    # YES/NO 파싱 (대소문자 무관, 공백 제거)
    df_model['Vote'] = df_model['Decision'].apply(lambda x: 1 if str(x).strip().upper().startswith('YES') else 0)
    
    if 'Simulation_Date' not in df_model.columns:
        # Static 모델인 경우 (Simulation_Date 컬럼 없음)
        return None
    
    # Dynamic 모델인 경우
    df_model['Simulation_Date'] = pd.to_datetime(df_model['Simulation_Date'])
    daily_ratio = df_model.groupby('Simulation_Date')['Vote'].mean().reset_index()
    daily_ratio.columns = ['Date', 'Purchase_Ratio']
    return daily_ratio.sort_values('Date')

def process_file(file_path, steam_gt, stock_gt):
    """
    개별 결과 파일을 처리하여 상관계수를 계산합니다.
    """
    try:
        model_df = pd.read_csv(file_path)
        
        # 일별 비율 계산
        dynamic_df = calculate_daily_ratio(model_df)
        
        if dynamic_df is None:
            # Static 모델은 시계열 상관분석 불가
            return {
                'File': os.path.basename(file_path),
                'Type': 'Static',
                'Correlation_Steam': np.nan,
                'P_Value_Steam': np.nan,
                'Correlation_Stock': np.nan,
                'P_Value_Stock': np.nan,
                'Num_Data_Points': 0,
                'Note': 'Static Model (No Time Series)'
            }
        
        # 1. Steam 데이터 병합 (Inner Join: 정확한 날짜 매칭)
        # Steam 데이터는 누락이 거의 없다고 가정하거나, 정확한 날짜에만 비교하는 것이 맞음
        merged_steam = pd.merge(steam_gt, dynamic_df, on='Date', how='inner')
        
        # 2. Stock 데이터 병합 (Merge Asof: 가장 가까운 날짜 매칭)
        # merge_asof를 사용하기 위해 양쪽 다 Date 기준 정렬이 되어 있어야 함
        stock_gt = stock_gt.sort_values('Date')
        dynamic_df = dynamic_df.sort_values('Date')
        
        # direction='nearest'로 가장 가까운 날짜의 종가를 가져옴
        merged_stock = pd.merge_asof(dynamic_df, stock_gt, on='Date', direction='nearest')
        
        # 데이터 포인트 부족 시 처리
        if len(merged_steam) < 2:
            return {
                'File': os.path.basename(file_path),
                'Type': 'Dynamic',
                'Correlation_Steam': np.nan,
                'P_Value_Steam': np.nan,
                'Correlation_Stock': np.nan,
                'P_Value_Stock': np.nan,
                'Num_Data_Points': len(merged_steam),
                'Note': 'Not enough overlap'
            }
            
        # Spearman 상관계수 계산
        # Steam
        corr_steam, p_steam = spearmanr(merged_steam['Purchase_Ratio'], merged_steam['Positive_Ratio'])
        
        # Stock
        # merge_asof 결과에서 NaN이 있을 수 있으므로 제거 (주가 데이터가 너무 멀리 떨어져 있어도 가져오지만, 
        # 만약 GT 데이터 범위 밖이라면 NaN일 수 있음)
        merged_stock = merged_stock.dropna(subset=['Stock_Price'])
        if len(merged_stock) < 2:
            corr_stock, p_stock = np.nan, np.nan
        else:
            corr_stock, p_stock = spearmanr(merged_stock['Purchase_Ratio'], merged_stock['Stock_Price'])
            
        return {
            'File': os.path.basename(file_path),
            'Type': 'Dynamic',
            'Correlation_Steam': corr_steam,
            'P_Value_Steam': p_steam,
            'Correlation_Stock': corr_stock,
            'P_Value_Stock': p_stock,
            'Num_Data_Points': len(merged_steam), # Steam 기준 매칭 수 (Stock은 다를 수 있음)
            'Note': 'Success'
        }
        
    except Exception as e:
        return {
            'File': os.path.basename(file_path),
            'Type': 'Error',
            'Correlation_Steam': np.nan,
            'P_Value_Steam': np.nan,
            'Correlation_Stock': np.nan,
            'P_Value_Stock': np.nan,
            'Num_Data_Points': 0,
            'Note': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Spearman Correlation Evaluation Script")
    parser.add_argument("--results_dir", type=str, default="results", help="결과 파일들이 있는 폴더 경로")
    parser.add_argument("--steam_gt", type=str, default="datasets/ground_truth_steam.csv", help="Steam GT 파일 경로")
    parser.add_argument("--stock_gt", type=str, default="datasets/ground_truth_stock.csv", help="Stock GT 파일 경로")
    parser.add_argument("--output_csv", type=str, default="results/result.csv", help="결과 요약 저장 경로")
    parser.add_argument("--period", type=str, choices=['daily', 'weekly'], default="daily", help="주가 데이터 기간 설정 (daily: 당일 종가, weekly: 7일 평균)")
    
    args = parser.parse_args()
    
    print(f"--- Spearman Correlation Evaluation ---")
    print(f"Target Directory: {args.results_dir}")
    print(f"Output File: {args.output_csv}")
    print(f"Period: {args.period}")
    
    # Ground Truth 로드
    try:
        steam_gt = load_ground_truth(args.steam_gt, 'Positive_Ratio')
        stock_gt = load_ground_truth(args.stock_gt, 'Stock_Price')
        
        if args.period == 'weekly':
            # 7일 이동 평균 계산 (데이터가 날짜순 정렬되어 있다고 가정 - load_ground_truth에서 정렬함)
            # min_periods=1로 설정하여 데이터 시작 부분도 계산되도록 함
            stock_gt['Stock_Price'] = stock_gt['Stock_Price'].rolling(window=7, min_periods=1).mean()
            print("   [Info] Stock Price: Calculated 7-day rolling average.")
    except Exception as e:
        print(f"[Critical Error] GT 데이터를 로드할 수 없습니다: {e}")
        return

    # 결과 파일 탐색
    if not os.path.exists(args.results_dir):
        print(f"[Error] 결과 폴더가 존재하지 않습니다: {args.results_dir}")
        return
        
    csv_files = glob.glob(os.path.join(args.results_dir, "*.csv"))
    # result.csv가 이미 있다면 제외 (자기 자신을 읽는 것 방지)
    csv_files = [f for f in csv_files if os.path.abspath(f) != os.path.abspath(args.output_csv)]
    
    if not csv_files:
        print("[Warning] 분석할 CSV 파일이 없습니다.")
        return
        
    print(f"총 {len(csv_files)}개의 파일을 발견했습니다. 분석을 시작합니다...")
    
    results = []
    for file_path in csv_files:
        print(f"Processing: {os.path.basename(file_path)}...")
        res = process_file(file_path, steam_gt, stock_gt)
        results.append(res)
        
    # 결과 요약 저장
    if results:
        df_results = pd.DataFrame(results)
        # 보기 좋게 컬럼 순서 정렬
        cols = ['File', 'Type', 'Correlation_Steam', 'P_Value_Steam', 'Correlation_Stock', 'P_Value_Stock', 'Num_Data_Points', 'Note']
        df_results = df_results[cols]
        
        df_results.to_csv(args.output_csv, index=False)
        print(f"\n[Done] 분석이 완료되었습니다. 결과가 저장되었습니다: {args.output_csv}")
        print("\n--- Summary ---")
        print(df_results[['File', 'Correlation_Steam', 'Correlation_Stock']])
    else:
        print("\n[Done] 처리된 결과가 없습니다.")

if __name__ == "__main__":
    main()
