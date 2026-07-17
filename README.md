# Edge-Native Vocab Tutor

This project is built as a practical showcase of the concepts and architectures learned from the [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) course. It is an edge-native, local-inference language learning backend designed to overcome the latency, cost, and structured output limitations of cloud-based LLMs.

## Table of Contents

- [Introduction: The Problem & The Solution](#introduction-the-problem--the-solution)
- [Key Problems Solved (Applied Learnings)](#key-problems-solved-applied-learnings)
- [Tech Stack & Architecture](#tech-stack--architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Endpoints Overview](#api-endpoints-overview)
- [Known Limitations (Scope of MVP)](#known-limitations-scope-of-mvp)

## Introduction: The Problem & The Solution

**1. Context & Challenges**:
While preparing for advanced English proficiency exams (GRE, TOEFL), I found that building a personalized vocabulary database is an extremely tedious process. It requires looking up context-specific definitions, example sentences, and derivatives, while being mindful of semantic nuances in translation. Although cloud-based LLMs (e.g., ChatGPT) can assist, they present three critical issues:
*   **Cost & Dependency**: Complete reliance on internet connectivity and recurring API token costs.
*   **Lack of Semantic Architecture**: Chat-based UIs lack automated mechanisms (like Vector Embeddings) to dynamically cluster personal vocabulary based on semantic similarity.
*   **Unstructured Output**: Pure conversational prompts struggle to provide the strict, structured data required for a programmatic learning loop (e.g., automated quiz generation).

**2. The Quantitative Reality**:
In traditional software like Anki or Quizlet, manually creating a high-quality flashcard—complete with synonyms, derived sentences, and custom quizzes—takes an average of 3 to 5 minutes. Building a foundational GRE deck of 1,000 words demands over 50 hours of pure manual labor. While public decks exist, they lack the personalization crucial for effective learning. Conversely, automating this via cloud APIs incurs constant network latency (~2-3 seconds per request) and perpetual billing.

**3. Why It's Worth Solving (The Impact)**:
To solve this, I implemented a **fully offline, Edge-Native LLM Learning System**, bridging my real-world needs with my learnings from the course. By leveraging model quantization (GGUF, 4-bit), the system successfully deploys an 8B-parameter model locally on a standard laptop. Combined with **Structured Outputs (Pydantic)** and an **in-memory Vector Store (Cosine Similarity)**, it guarantees stable, personalized vocabulary clustering and quiz generation—achieving zero API costs, zero network latency, and complete data privacy.

## Key Problems Solved (Applied Learnings)

This project translates the theoretical and architectural concepts from the `ai-engineering-from-scratch` curriculum into practical engineering solutions:

1. **Edge Inference & Model Quantization (Phase 17)**:
   To overcome hardware limitations and eliminate API costs, the system utilizes a 4-bit quantized GGUF model (`Llama-3-8B-Instruct-Q4_K_M`). This enables offline, low-latency execution on consumer hardware while ensuring 100% data privacy.
2. **Structured Outputs & Constrained Decoding (Phase 11)**:
   Cloud LLMs natively struggle with consistent JSON formatting. By integrating Pydantic schemas with the constrained decoding capabilities of `llama-cpp-python`, the system guarantees strictly typed JSON responses, preventing downstream parsing failures.
3. **Embeddings & Semantic Clustering (Phase 11)**:
   Addressing the "vocabulary mismatch" problem in traditional tag-based categorization. The system implements a lightweight in-memory vector store (SQLite + `SentenceTransformers`) and calculates Cosine Similarity to dynamically cluster semantically related words.
4. **Prompt Engineering & Input Sanitization (Phase 11)**:
   To prevent delimiter collision and prompt injection attacks from user-provided sentences, the system employs **XML tag sandboxing** (e.g., `<user_sentence>`). Combined with precise Role Prompting and Negative Constraints, it successfully anchors the model's behavior to an objective evaluator persona.
5. **Chain-of-Thought (CoT) Evaluator (Phase 11)**:
   Small Language Models (SLMs) are prone to hallucinated over-corrections during assessment. The evaluation endpoint enforces a CoT pipeline, compelling the model to explicitly output `reasoning` steps prior to rendering a final boolean judgment, drastically improving grading accuracy.

## Tech Stack & Architecture

* **LLM Engine**: `llama-cpp-python`
* **Model**: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (4-bit Quantization)
* **Structured Data Validation**: `Pydantic`
* **Embedding Model**: `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **Vector Storage & Retrieval**: `SQLite3` + `NumPy` (In-memory Cosine Similarity computation)
* **Backend API**: `FastAPI` + `Uvicorn` (Asynchronous REST API)
* **Frontend Visualization**: `Streamlit` (Still in progress)

## Project Structure

```text
edge-vocab-tutor/
├── models/                 # Directory for downloaded models
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI application
│   ├── db/
│   │   └── database.py     # Cosine similarity logic
│   └── llm/
│       ├── generator.py    # Structured output generation
│       └── quiz.py         # CoT Sentence Evaluator
├── requirements.txt        # Project dependencies
├── .gitignore
└── README.md
```

## Installation & Setup

**1. Prerequisites**

* Python 3.10+
* `uv` package manager (recommended)
* Git, wget, build-essential (for WSL/Ubuntu)

**2. Clone the repository & Install dependencies**

```bash
git clone https://github.com/yourusername/edge-vocab-tutor.git
cd edge-vocab-tutor
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
*(Note: To enable GPU acceleration on NVIDIA devices, install `llama-cpp-python` with `CMAKE_ARGS="-DGGML_CUDA=on"`)*

**3. Download the Quantized Model**

```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## Running the Application

This system is decoupled into a backend API and a frontend dashboard.

**1. Start the FastAPI Backend**

Run the ASGI server to load the models into memory (loads as a singleton to prevent OOM errors):

```bash
uvicorn src.api.main:app --reload
```
API Documentation (Swagger UI) will be available at: `http://127.0.0.1:8000/docs`

## API Endpoints Overview

* `POST /api/generate`: Accepts a target word and context, generates a detailed vocabulary card (CEFR, Senses, Synonyms, Antonyms) in strictly typed JSON.
* `GET /api/similar/{word}`: Retrieves the top 5 most semantically similar words stored in the local SQLite database.
* `POST /api/evaluate`: Accepts a user-generated sentence, runs a CoT prompt, and returns structured feedback.

## Known Limitations (Scope of MVP)

* **Model Constraints**: Relying on an 8B quantized model may lead to over-correction in complex linguistic edge cases during sentence evaluation.
* **Vector Store Scalability**: Vector similarity is computed in-memory via NumPy. This is highly efficient for vocabulary sets (< 10,000 items) but would require transitioning to an HNSW-based database (e.g., Qdrant or Milvus) if scaled to million-document enterprise retrieval.