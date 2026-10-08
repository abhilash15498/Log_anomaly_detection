import time
import warnings
warnings.filterwarnings("ignore")
from src.detection.infer import detect

def run_demo():
    print("=" * 60)
    print("=== AI-Driven Infrastructure Monitoring - Detection Demo ===")
    print("=" * 60)
    
    print("\n[1/3] Loading pre-processed log data and trained models...")
    start_time = time.time()
    
    # We load a subset of 1000 logs just to make the demo run instantly, 
    # but you can remove limit=1000 to process the entire dataset.
    results = detect(limit=1000) 
    
    elapsed = time.time() - start_time
    print(f"      Done in {elapsed:.2f} seconds.\n")
    
    # Count anomalies
    total_logs = len(results)
    anomalous_logs = [r for r in results if r.is_anomaly]
    anomaly_count = len(anomalous_logs)
    
    print(f"[2/3] Analyzing results...")
    print(f"      Total Logs Analyzed : {total_logs}")
    print(f"      Anomalies Detected  : {anomaly_count}")
    print(f"      Anomaly Rate        : {(anomaly_count/total_logs)*100:.2f}%\n")
    
    print("[3/3] Displaying Top 3 Anomalous Events:")
    print("-" * 60)
    
    # Sort anomalies by score (highest first) to show the most critical ones
    anomalous_logs.sort(key=lambda x: x.anomaly_score, reverse=True)
    
    for i, anomaly in enumerate(anomalous_logs[:3]):
        print(f"[!] Anomaly #{i+1}")
        print(f"   Log ID    : {anomaly.log_id}")
        print(f"   Severity  : {anomaly.anomaly_score:.4f} (Threshold: {anomaly.threshold_used:.4f})")
        print(f"   Context   :")
        for line in anomaly.context_window:
            # Highlight the actual anomaly line
            prefix = "      >> " if line == anomaly.context_window[len(anomaly.context_window)//2] else "         "
            print(f"{prefix}{line}")
        print("-" * 60)

if __name__ == "__main__":
    run_demo()
