import json
import requests
import time
from datetime import datetime

API_EVAL_URL = "http://127.0.0.1:8000/api/evaluate"
DATASET_PATH = "tests/eval_dataset.json"
REPORT_PATH = "tests/eval_report.md"

def load_dataset():
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def run_evaluation():
    dataset = load_dataset()
    if not dataset:
        return

    total_cases = len(dataset)

    metrics = {
        "json_parse_errors": 0,    # times of returning 500
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "nonsense_rejected": 0,
        "nonsense_failed": 0
    }

    report_details = []

    print(f"Activate evaluate pipeline for {total_cases} cases...")

    for idx, item in enumerate(dataset, 1):
        print(f"[{idx}/{total_cases}] Category: {item['category']}...")

        payload = {
            "word": item["word"],
            "definition": item["definition"],
            "user_sentence": item["user_sentence"]
        }

        start_time = time.time()
        try:
            res = requests.post(API_EVAL_URL, json=payload, timeout=30)
            latency = time.time() - start_time

            if res.status_code != 200:
                metrics["json_parse_errors"] += 1
                report_details.append(format_log(item, "JSOM PARSE ERROR", "N/A", latency))
                continue

            data = res.json()
            model_judgement = data.get("is_appropriate", False)
            expected = item["expected_is_appropriate"]
            category = item["category"]

            status = "UNKNOWN"
            if "nonsense" in category:
                if not model_judgement:
                    metrics["nonsense_rejected"] += 1
                    status = "Rejected nonsence successfully"
                else:
                    metrics["nonsense_failed"] += 1
                    status = "Failed to reject nonsence"
            else:
                if expected == True and model_judgement == True:
                    metrics["true_positive"] += 1
                    status = "TP"
                elif expected == False and model_judgement == False:
                    metrics["true_negative"] += 1
                    status = "TN"
                elif expected == True and model_judgement == False:
                    metrics["false_positive"] += 1
                    status = "FP"
                elif expected == False and model_judgement == True:
                    metrics["false_negative"] += 1
                    status = "FN"

            report_details.append(
                format_log(item, status, f"**Reasoning:** {data.get('reasoning')}\n\n**Feedback:** {data.get('feedback')}", latency)
            )
        except Exception as e:
            metrics["json_parse_errors"] += 1
            report_details.append(format_log(item, f"REQUEST EXCEPTION: {e}", "N/A", 0))

    generate_markdown_report(metrics, total_cases, report_details)

def format_log(item, status, model_output, latency):
    return (
        f"### {item['word']} ({item['category']})\n"
        f"- **Sentence:** `{item['user_sentence']}`\n"
        f"- **Status:** {status} | **Latency:** {latency:.2f}s\n"
        f"- **Expected Valid:** {item['expected_is_appropriate']}\n\n"
        f"{model_output}\n"
        "---\n"
    )

def generate_markdown_report(m, total, details):
    successful_parses = total - m["json_parse_errors"]
    json_success_rate = (successful_parses / total) * 100 if total > 0 else 0

    expected_true = m["true_positive"] + m["false_positive"]
    expected_false = m["true_negative"] + m["false_negative"]
    nonsense = m["nonsense_rejected"] + m["nonsense_failed"]

    fp_rate = (m["false_positive"] / expected_true * 100) if expected_true > 0 else 0
    fn_rate = (m["false_negative"] / expected_false * 100) if expected_false > 0 else 0
    nonsense_rej_rate = (m["nonsense_rejected"] / nonsense * 100) if nonsense > 0 else 0

    report = (
        f"# Evaluation Report\n"
        f"*Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        f"## KPIs\n"
        f"1. **JSON Parsing Success Rate**: `{json_success_rate:.1f}%` (Target: ≥95%)\n"
        f"2. **False Positive Rate**: `{fp_rate:.1f}%` (Target: ≤20%)\n"
        f"3. **False Negative Rate**: `{fn_rate:.1f}%` (Target: ≤5%)\n"
        f"4. **Nonsense Rejection Rate**: `{nonsense_rej_rate:.1f}%` (Target: ≥90%)\n\n"
        f"## Raw Metrics\n"
        f"- Total Cases: {total}\n"
        f"- Parse Errors: {m['json_parse_errors']}\n"
        f"- True Positive (TP): {m['true_positive']}\n"
        f"- True Negative (TN): {m['true_negative']}\n"
        f"- False Positive (FP): {m['false_positive']}\n"
        f"- False Negative (FN): {m['false_negative']}\n\n"
        f"## Detailed Logs (Feedback)\n\n"
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report + "\n".join(details))

    print(f"\nEvaluation pipeline complete.\nReport formed at \"{REPORT_PATH}\"")
    print("--------------------------------------------------")
    print(f"JSON Success Rate    : {json_success_rate:.1f}%")
    print(f"False-Positive Rate  : {fp_rate:.1f}%")
    print(f"False-Negative Rate  : {fn_rate:.1f}%")
    print(f"Nonsence Reject Rate : {nonsense_rej_rate:.1f}%")

if __name__ == "__main__":
    run_evaluation()