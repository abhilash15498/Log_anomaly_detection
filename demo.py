import os
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.is_file():
    load_dotenv(dotenv_path=_env_file)

from src.detection.infer import detect
from src.llm.diagnose import diagnose_anomaly

def run_demo():
    print("\n" + "=" * 65)
    print("      AI INFRASTRUCTURE MONITORING & INCIDENT DIAGNOSIS")
    print("=" * 65)

    print("\n[+] Scanning system logs with ML detection ensemble...")
    start_time = time.time()
    
    # Process sample logs
    results = detect(limit=1000) 
    elapsed = time.time() - start_time
    
    anomalous_logs = [r for r in results if r.is_anomaly]
    print(f"    Scanned {len(results):,} logs in {elapsed:.2f}s | Flagged {len(anomalous_logs)} critical incidents\n")
    
    print("=" * 65)
    print("                      TOP INCIDENT REPORTS")
    print("=" * 65)
    
    anomalous_logs.sort(key=lambda x: x.anomaly_score, reverse=True)
    
    for i, anomaly in enumerate(anomalous_logs[:3], 1):
        report = diagnose_anomaly(anomaly)

        print(f"\n[INCIDENT #{i}] - ID: {anomaly.log_id}")
        print(f"Severity    : {report.severity.upper()}")
        print(f"Diagnosis   : {report.explanation}")
        
        print("\nRelevant Log Sequence:")
        mid_idx = len(anomaly.context_window) // 2
        for idx, line in enumerate(anomaly.context_window):
            prefix = "  >> " if idx == mid_idx else "     "
            print(f"{prefix}{line}")
        
        print("\nRecommended Fix:")
        for step_num, action in enumerate(report.recommended_actions, 1):
            print(f"  {step_num}. {action}")
        print("-" * 65)

if __name__ == "__main__":
    run_demo()
