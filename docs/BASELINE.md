# Profiling Report: Baseline

This document defines the performance baseline for the current
Edge-Vocab-Tutor system. All subsequent methods and experiments
are evaluated relative to this configuration unless otherwise specified.

> **Generated at**: 2026-09-12 16:59:22
> **Source Data**: `experiments/baseline/baseline.json`
> **Tags**: baseline, sequential, api-level, llama-3-8b-q4

## 1. Environment Configuration
- **Model**: Meta-Llama-3-8B-Instruct-Q4_K_M.gguf (Q4_K_M)
- **Hardware**: NVIDIA GeForce RTX 5070 Ti
- **VRAM Usage**: 5131 MiB

## 2. Workload Parameters
- **Type**: Single-request sequential inference
- **Concurrency**: 1
- **Warm-up**: 1
- **Measured iterations**: 10

## 3. Metrics Summary
### End-to-End Latency
- **Mean**: 1445.29 ms
- **P50**: 1403.90 ms
- **P95**: 1714.82 ms

### LLM Inference
- **Mean TTFT**: 61.64 ms
- **Mean TPOT**: 17.31 ms

## 4. Raw Runs Data
| Run ID | Cold Start | Queue (ms) | TTFT (ms) | Output Tokens | TPOT (ms) | E2E (ms) |
|--------|------------|------------|-----------|---------------|-----------|----------|
| 1 | Yes | 0.00 | 491.99 | 80 | 18.26 | 1952.10 |
| 2 | No | 0.00 | 13.93 | 80 | 17.55 | 1417.70 |
| 3 | No | 0.00 | 14.25 | 80 | 16.97 | 1371.00 |
| 4 | No | 0.00 | 13.36 | 80 | 17.33 | 1399.40 |
| 5 | No | 0.00 | 13.49 | 80 | 17.39 | 1404.40 |
| 6 | No | 0.00 | 14.46 | 80 | 17.58 | 1420.10 |
| 7 | No | 0.00 | 13.83 | 80 | 17.65 | 1424.80 |
| 8 | No | 0.00 | 14.25 | 80 | 17.38 | 1403.40 |
| 9 | No | 0.00 | 13.91 | 80 | 16.46 | 1329.10 |
| 10 | No | 0.00 | 12.97 | 80 | 16.49 | 1330.90 |
