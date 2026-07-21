import requests
import time
import statistics

API_BASE = "http://127.0.0.1:8000/api"
ITERATIONS = 100

def run_test(name, method, url, payload, log_filename):
    success_count = 0
    latencies = []

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"=== Stability Test for {name} ({ITERATIONS} runs) ===\n\n")
        print(f"Start testing [{name}] ... ({ITERATIONS} times)")

        for i in range(ITERATIONS):
            try:
                start_time = time.time()
                if method == "POST":
                    res = requests.post(url, json=payload)
                else:
                    res = requests.get(url)
                latency = time.time() - start_time

                if res.status_code == 200:
                    success_count += 1
                    latencies.append(latency)
                    log_line = f"[Run {i+1:03d}] SUCCESS | Latency: {latency:.4f}s\n"
                else:
                    log_line = f"[Run {i+1:03d}] FAILED  | Status: {res.status_code} | Error: {res.text}\n"

                f.write(log_line)

                if (i + 1) % 10 == 0:
                    print(f"  -> Progress: {i+1}/{ITERATIONS}")

            except Exception as e:
                f.write(f"[Run {i+1:03d}] ERROR   | Exception: {str(e)}\n")

        avg_lat = statistics.mean(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0
        min_lat = min(latencies) if latencies else 0.0
        success_rate = (success_count / ITERATIONS) * 100

        summary = (
            f"\n=== Summary for {name} ===\n"
            f"Success Rate: {success_rate:.1f}% ({success_count}/{ITERATIONS})\n"
            f"Avg Latency: {avg_lat:.4f} s\n"
            f"Min Latency: {min_lat:.4f} s\n"
            f"Max Latency: {max_lat:.4f} s\n"
            f"----------------------------------------\n"
        )
        f.write(summary)
        print(summary)

if __name__ == "__main__":
    print(f"Start Local LLM testing, {ITERATIONS * 3} calls...\n")

    run_test(
        name="Generate Vocab Card",
        method="POST",
        url=f"{API_BASE}/generate",
        payload={"word": "paradigm"},
        log_filename="log_generate.txt"
    )

    run_test(
        name="Semantic Similarity Search",
        method="GET",
        url=f"{API_BASE}/similar/paradigm?top_k=5",
        payload=None,
        log_filename="log_similar.txt"
    )

    run_test(
        name="CoT Sentence Evaluation",
        method="POST",
        url=f"{API_BASE}/evaluate",
        payload={
            "word": "paradigm",
            "definition": "A typical example or pattern of something; a model.",
            "user_sentence": "The discovery of DNA created a new paradigm in biological science."
        },
        log_filename="log_evaluate.txt"
    )

    print("\n✅ All tests complete. Outputs in log_generate.txt, log_similar.txt, log_evaluate.txt")