# Edge-LLM-Research

> **Research Status:** Exploratory / Baseline Characterization
>
> The research direction is currently exploratory. Multiple
> systems-level directions, including inference optimization,
> retrieval, and resource-aware execution, are being investigated.
> No final research method has been selected yet.


An empirical systems research project investigating the inference characteristics and bottlenecks of LLMs deployed in resource-constrained local environments.

## 1. Project Paradigm & Methodology

This repository originated as an MVP for a local-inference vocabulary tutor. It has since evolved into an **experimental research project**.

Instead of treating the LLM as a generic black-box API, this project utilizes the vocabulary tutoring application (which involves constrained decoding, RAG, and structured outputs) as a **stable, realistic application workload**. By constraining the model's output variability (achieving 100% JSON parsing success), we establish a more controlled and reproducible workload for evaluating underlying system performance—such as Time-to-First-Token (TTFT), Time-Per-Output-Token (TPOT), and Queueing Latency.

## 2. Repository Architecture

The repository strictly separates the application workload from the experimental instrumentation and future research methods:

```text
.
├── src/                    # The Application Workload (FastAPI, SQLite, Llama.cpp)
│   ├── api/                # HTTP layer and concurrency control (Mutex locks)
│   ├── db/                 # Vector retrieval and persistent storage
│   └── llm/                # Structured output generation and CoT prompting
├── experiments/            # Core Systems Research & Profiling
│   ├── baseline/           # Foundation metrics (E2E latency, Success rates)
│   └── output_length/      # Active: Linear characterization of TPOT vs. Output Tokens
├── methods/                # (Planned) Future implementations of routing, caching, or optimization algorithms
├── docs/                   # Documentation and historical artifacts
│   ├── BASELINE.md         # Baseline system metrics summary
│   └── MVP_EVALUATION.md   # Record of prompt tuning used to stabilize the workload output
├── tests/                  # Legacy MVP evaluation and testing scripts
└── requirements.txt
```

## 3. Current Experiments

The research phase is actively ongoing in the `experiments/` directory.

### 3.1. Workload Stabilization & Baseline (`docs/MVP_EVALUATION.md` & `experiments/baseline`)
Before profiling the system, the application workload was stabilized. Multi-turn Few-Shot CoT prompting and regex-based robust parsing were implemented to achieve a 100% JSON parse success rate. This ensures output length predictability. A baseline was then established for application-level E2E latency and basic queuing behavior on consumer hardware (RTX 5070 Ti, 16GB VRAM).

### 3.2. Output Length Scaling (`experiments/output_length`) - *Active*
A controlled one-factor-at-a-time (OFAT) experiment examining the relationship between actual output length and decoding latency.
Current observations show that decode time exhibits a strong linear relationship with output length, while TPOT shows a positive but weaker trend.


## 4. Environment Setup & Reproducibility

**Prerequisites:**
- Python 3.10+
- `uv` package manager

**Installation:**
```bash
git clone https://github.com/yourusername/Edge-LLM-Research.git
cd Edge-LLM-Research
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Note: For NVIDIA GPU acceleration, compile llama-cpp-python with CUDA:
# CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python
```

**Model Acquisition:**
The current experiments are standardized on Llama-3-8B (Q4_K_M).
```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## 5. Running Experiments

To execute the current active benchmark (Output Length Characterization):
```bash
# Run the benchmark script to generate raw JSON data
python experiments/output_length/output_length.py

# Generate statistical plots and linear regression analysis (requires matplotlib, numpy)
python experiments/output_length/plot_output_length.py results/profiling/YOUR_RESULT_FILE.json
```