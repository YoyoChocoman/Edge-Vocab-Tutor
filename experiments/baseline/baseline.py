import requests
import time
import statistics
import json

API_URL = "http://127.0.0.1:8000/api/evaluate"
ITERATIONS = 10

payload = {
    "word": "mitigate",
    "definition": "Make a situation less severe, harmful, or painful",
    "user_sentence": "The government implemented new flood defenses to mitigate the flood caused by heavy rains."
}

def run_baseline():
    print(f"Starting Baseline Profiling (Sequential, N={ITERATIONS})...")
    e2e_latencies = []

    for i in range(ITERATIONS):
        req_type = "Cold Start" if i == 0 else "Warm"
        print(f"Run {i+1:02d} ({req_type})...", end=" ", flush=True)

        t_start = time.perf_counter()
        try:
            res = requests.post(API_URL, json=payload, timeout=30)
            t_end = time.perf_counter()

            if res.status_code == 200:
                e2e = t_end - t_start
                e2e_latencies.append(e2e)
                print(f"Success | E2E: {e2e:.4f}s")
            else:
                print(f"Failed | Status: {res.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Error | {e}")

        time.sleep(1)

    if not e2e_latencies:
        print("No successful requests to analyze.")
        return

    e2e_latencies.sort()

    def percentile(data, p):
        idx = (len(data) - 1) * p
        lower = int(idx)
        upper = lower + 1 if lower + 1 < len(data) else lower
        weight = idx - lower
        return data[lower] * (1 - weight) + data[upper] * weight

    p50 = percentile(e2e_latencies, 0.50)
    p95 = percentile(e2e_latencies, 0.95)
    mean = statistics.mean(e2e_latencies)

    print("\n=== Application E2E Latency Summary ===")
    print(f"Iterations : {len(e2e_latencies)}")
    print(f"Min        : {e2e_latencies[0]:.4f}s")
    print(f"Mean       : {mean:.4f}s")
    print(f"P50 (Med)  : {p50:.4f}s")
    print(f"P95        : {p95:.4f}s")
    print(f"Max        : {e2e_latencies[-1]:.4f}s")
    print("\n* LLM Tier details (TTFT/TPOT) can be observed in the FastAPI terminal log.")

if __name__ == "__main__":
    run_baseline()