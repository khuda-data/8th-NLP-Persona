"""
모든 분석 스크립트 실행
"""
import subprocess
import sys
import os

def run_script(script_name):
    """스크립트 실행"""
    print(f"\n{'='*70}")
    print(f"실행 중: {script_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
            capture_output=False
        )
        print(f"\n✅ {script_name} 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} 실패: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("실험 검증 분석 전체 실행")
    print("="*70)
    
    scripts = [
        "analyze_time_decay_effect.py",
        "analyze_persona_differences.py",
        "compare_team2_team3.py",
        "compare_all_methods.py"
    ]
    
    results = []
    for script in scripts:
        if os.path.exists(script):
            success = run_script(script)
            results.append((script, success))
        else:
            print(f"\n⚠️  {script} 파일을 찾을 수 없습니다.")
            results.append((script, False))
    
    # 결과 요약
    print("\n" + "="*70)
    print("실행 결과 요약")
    print("="*70)
    for script, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {script:40s} {status}")
    
    print("\n" + "="*70)
    print("✅ 모든 분석 완료!")
    print("="*70)
    print("\n📁 결과 파일 위치:")
    print("  - figures/ : 그래프")
    print("  - results/ : 통계 결과")

