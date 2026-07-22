# Edge-Native Vocab Tutor

This project is built as a practical showcase of the concepts and architectures learned from the [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) course. It is an edge-native, local-inference language learning backend designed to overcome the latency, cost, structured output limitations, and privacy risks of cloud-based LLMs.

## Table of Contents
- [Introduction: The Problem & The Solution](#introduction-the-problem--the-solution)
- [Key Architectural Implementations](#key-architectural-implementations)
- [Tech Stack & Architecture](#tech-stack--architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Evaluation & Trade-offs](EVALUATION.md) *(See detailed ML metrics)*

## Introduction: The Problem & The Solution

**1. Context & Challenges**
Building a personalized vocabulary database for advanced English proficiency exams (e.g., GRE, TOEFL) is highly tedious. While cloud-based LLMs (e.g., ChatGPT) can assist, they introduce critical engineering and privacy issues:
*   **Cost & Privacy**: Complete reliance on external APIs incurs recurring token costs and exposes user data to the cloud.
*   **Lack of Semantic Architecture**: Traditional chat UIs lack automated mechanisms (e.g., Vector Embeddings) to dynamically cluster personal vocabulary based on semantic similarity.
*   **Unstructured Output**: Conversational prompts struggle to provide the strict, structured JSON required for programmatic learning loops.

**2. The Impact (Solution)**
To resolve this, I engineered a **fully offline, Edge-Native LLM Learning System**. By leveraging 4-bit model quantization (GGUF), the system successfully deploys an 8B-parameter model locally with a minimal VRAM footprint (~5GB). Combined with **Structured Outputs (Pydantic)** and an **in-memory Vector Store (Cosine Similarity)**, it guarantees stable, personalized vocabulary clustering and quiz generation—achieving zero API costs, zero network dependency, and complete data privacy.

## Key Architectural Implementations

1. **Edge Inference & Model Quantization (Phase 17)**:
   Deployed `Llama-3-8B-Instruct-Q4_K_M` to execute on consumer hardware. The 4-bit quantization effectively reduces memory footprint while maintaining reasoning integrity.
2. **Robust Parsing & Constrained Decoding (Phase 11)**:
   Initial stress tests showed a 23% JSON parsing failure rate due to uncontrolled markdown generation by the 8B model. Implemented regex-based JSON extraction and relaxed parsing (`strict=False`) to elevate API response stability to **100%**.
3. **Embeddings & Semantic Clustering (Phase 11)**:
   Resolved the "vocabulary mismatch" problem by building an in-memory vector store (SQLite + `SentenceTransformers`). Computes Cosine Similarity dynamically to aggregate semantically related vocabulary without relying on heavy external vector databases.
4. **Input Sanitization & XML Sandboxing (Phase 11)**:
   Migrated from traditional quote delimiters to **XML tag sandboxing** (e.g., `<user_sentence>`). This effectively sanitizes user inputs, preventing delimiter collisions and mitigating Prompt Injection attacks.
5. **Multi-Turn Few-Shot CoT Evaluator (Phase 11)**:
   Mitigated the SLM's (Small Language Model) tendency to hallucinate non-existent grammatical errors by evolving the prompt from Zero-Shot to **Few-Shot Chain-of-Thought**. For details on error rate reduction (False Positive/Negative), refer to the [Evaluation Report](EVALUATION.md).

## Tech Stack & Architecture

* **LLM Engine**: `llama-cpp-python` (C++ backend for optimized CPU/GPU edge inference)
* **Model**: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (4-bit Quantization)
* **Structured Data Validation**: `Pydantic`
* **Embedding Model**: `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **Vector Storage**: `SQLite3` + `NumPy` (In-memory Cosine Similarity)
* **Backend API**: `FastAPI` + `Uvicorn` (Asynchronous REST API with Mutex Lock for VRAM protection)
* **Frontend UI**: `Streamlit`

## Project Structure

```text
edge-vocab-tutor/
├── models/                 # Directory for downloaded models
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI application entry point
│   ├── db/
│   │   └── database.py     # Cosine similarity logic & DB
│   └── llm/
│       ├── generator.py    # Structured output generation
│       └── quiz.py         # Few-shot CoT Sentence Evaluator
├── tests/
│   ├── eval_dataset.json   # Golden dataset for evaluation
│   ├── run_eval.py         # Automated FP/FN metric script
│   └── test_load.py        # End-to-end API stress test
├── requirements.txt
├── .gitignore
├── EVALUATION.md           # Model performance and analysis
└── README.md
```

## Installation & Setup

**1. Prerequisites**
* Python 3.10+
* `uv` package manager (recommended)
* Git, wget, build-essential (for WSL/Ubuntu environments)

**2. Clone and Install**
```bash
git clone https://github.com/yourusername/edge-vocab-tutor.git
cd edge-vocab-tutor
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
*(Note: To enable GPU acceleration on NVIDIA devices, use `CMAKE_ARGS="-DGGML_CUDA=on"` during `llama-cpp-python` installation)*

**3. Download the Quantized Model**
```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## Running the Application

**1. Start the FastAPI Backend**
```bash
uvicorn src.api.main:app --reload
```
*Swagger UI available at: `http://127.0.0.1:8000/docs`*

**2. Start the Streamlit Frontend (In a separate terminal)**
```bash
source .venv/bin/activate
streamlit run src/ui/app.py
```