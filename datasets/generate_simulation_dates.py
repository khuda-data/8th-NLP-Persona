import pandas as pd
import os

def generate_simulation_dates():
    # 저장 경로 설정
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "simulation_dates.csv")

    dates_data = []

    # ==========================================
    # 1. 고정 이벤트 (Key Events) - 무조건 포함
    # ==========================================
    key_events = {
        "2020-12-10": "Game Release (Launch Day)",
        "2020-12-18": "Sony Removes CP2077 from Store",
        "2021-01-13": "CDPR Official Apology Video",
        "2021-06-21": "Return to PlayStation Store",
        "2022-02-15": "Patch 1.5 (Next-Gen Update)",
        "2022-09-13": "Edgerunners Anime Release (Trigger)",
        "2022-09-25": "Edgerunners Hype Peak",
        "2023-09-26": "Phantom Liberty DLC Release"
    }
    
    for date, desc in key_events.items():
        dates_data.append({"date": date, "description": desc, "type": "event"})

    # ==========================================
    # 2. 초기 격동기 (Weekly) - 출시 후 약 3개월
    # ==========================================
    # 2020-12-10 부터 2021-02-28 까지 매주 확인
    weekly_dates = pd.date_range(start="2020-12-10", end="2021-02-28", freq="W-THU")
    
    for dt in weekly_dates:
        date_str = dt.strftime("%Y-%m-%d")
        # 이미 이벤트에 있으면 중복 방지
        if date_str not in key_events:
            dates_data.append({"date": date_str, "description": "Weekly Monitoring (Volatile Phase)", "type": "weekly"})

    # ==========================================
    # 3. 안정기 (Monthly) - 2021.03 ~ 2023.12
    # ==========================================
    # 매월 1일 확인
    monthly_dates = pd.date_range(start="2021-03-01", end="2024-08-01", freq="MS") # MS: Month Start
    
    for dt in monthly_dates:
        date_str = dt.strftime("%Y-%m-%d")
        # 주요 이벤트 날짜와 겹치거나 너무 가까우면(3일 이내) 제외할 수도 있으나, 여기선 단순 중복만 체크
        if date_str not in key_events:
            dates_data.append({"date": date_str, "description": "Monthly Monitoring (Stable Phase)", "type": "monthly"})

    # ==========================================
    # 4. 데이터 프레임 생성 및 정렬
    # ==========================================
    df = pd.DataFrame(dates_data)
    
    # 날짜 기준 오름차순 정렬
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # 중복 날짜 제거 (혹시 겹쳤다면 이벤트 설명을 우선하기 위해 keep='first' 대신 로직 필요하지만, 위에서 처리함)
    df = df.drop_duplicates(subset=['date'], keep='first')

    # 다시 문자열로 변환
    df['date'] = df['date'].dt.strftime("%Y-%m-%d")

    # ==========================================
    # 5. CSV 저장
    # ==========================================
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Simulation dates generated: {len(df)} points.")
    print(f"📂 Saved to: {output_file}")
    
    # 미리보기 출력
    print("\n[Preview]")
    print(df[['date', 'description']].head())
    print("...")
    print(df[['date', 'description']].tail())

if __name__ == "__main__":
    generate_simulation_dates()