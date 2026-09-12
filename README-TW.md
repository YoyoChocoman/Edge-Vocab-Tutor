# Edge-LLM-Research

>研究狀態： 探索階段／基線特徵分析
>
>目前的研究方向仍處於探索階段。現階段正在研究多個系統層面的方向，包括推論最佳化（Inference Optimization）、檢索（Retrieval）以及資源感知執行（Resource-Aware Execution）等。目前尚未確定最終的研究方法。

一項實證系統研究項目，旨在探討部署於資源受限本地環境中的大語言模型（LLM）的推理特性與瓶頸。

## 1. 專案定位與研究方法

本 Repository 最初是一個用於本地推論（Local Inference）單字學習工具的 MVP，後續逐漸發展為一個實驗性系統研究專案。

本專案並不將 LLM 視為一般的黑盒 API，而是將單字教學應用程式作為一個穩定且具現實代表性的應用工作負載（Application Workload）。

該應用程式包含受限解碼（Constrained Decoding）、**檢索增強生成（Retrieval-Augmented Generation, RAG）以及結構化輸出（Structured Outputs）**等特性。透過限制模型輸出的變異性，並達成 100% JSON 解析成功率，可以建立一個更加受控且可重現的工作負載，進而評估底層系統效能，例如：

Time-to-First-Token（TTFT）
Time-Per-Output-Token（TPOT）
Queueing Latency（排隊延遲）

這種方法能夠降低應用層輸出不穩定所造成的干擾，使研究能更聚焦於 LLM 推論系統本身的效能特徵與系統瓶頸。

## 2. Repository 架構

本 Repository 嚴格將應用程式工作負載與**實驗量測工具（Experimental Instrumentation）以及未來預計實作的研究方法（Research Methods）**分離：

```text
.
├── src/                    # 應用程式工作負載（FastAPI、SQLite、Llama.cpp）
│   ├── api/                # HTTP 層與並行控制（Mutex Locks）
│   ├── db/                 # 向量檢索與持久化儲存
│   └── llm/                # 結構化輸出生成與 CoT Prompting
├── experiments/            # 核心系統研究與效能分析
│   ├── baseline/           # 基礎指標（E2E Latency、成功率）
│   └── output_length/      # 進行中：TPOT 與輸出 Token 數量的線性特徵分析
├── methods/                # （規劃中）未來的路由、快取或最佳化演算法
├── docs/                   # 文件與歷史研究紀錄
│   ├── BASELINE.md         # 基線系統指標摘要
│   └── MVP_EVALUATION.md   # 用於穩定工作負載輸出的 Prompt 調整紀錄
├── tests/                  # 舊版 MVP 評估與測試腳本
└── requirements.txt
```

## 3. 目前進行中的實驗

目前的研究階段主要在 experiments/ 目錄中進行。

### 3.1. 工作負載穩定化與基線建立（docs/MVP_EVALUATION.md & experiments/baseline）

在進行系統效能分析之前，首先對應用程式工作負載進行穩定化。
透過實作多輪 Few-Shot CoT Prompting以及基於 Regex 的穩健解析機制（Robust Parsing），使模型達成 100% JSON 解析成功率。
這項工作能夠提高輸出格式的一致性，並使輸出長度更加可預測，降低工作負載本身的變異性。
在此基礎上，建立系統的基線效能，主要量測：應用程式層級的 E2E Latency、基本的 Queueing Behavior、消費級硬體上的推論效能

目前使用的硬體環境為：
GPU：NVIDIA RTX 5070 Ti、VRAM：16 GB

### 3.2. 輸出長度擴展特性分析（experiments/output_length）— 進行中

目前正在進行一項受控的單因子實驗（One-Factor-at-a-Time, OFAT），用於分析實際輸出長度（Output Length）與解碼延遲（Decoding Latency）之間的關係。
目前的初步觀察顯示：Decode Time 與輸出長度之間呈現強烈的線性關係且TPOT 與輸出長度之間也呈現正向趨勢，但相關程度較弱。
目前此實驗主要用於建立 LLM 推論在不同輸出長度下的基礎效能特徵（Performance Characterization），後續研究將以此基線作為比較不同最佳化方法的基礎。

## 4. 環境設定與可重現性
**系統需求:**
- Python 3.10+
- uv 套件管理器

**安裝:**
```bash
git clone https://github.com/yourusername/Edge-LLM-Research.git
cd Edge-LLM-Research
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# 注意：若要使用 NVIDIA GPU 加速，
# 請使用 CUDA 編譯 llama-cpp-python：CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python
```

**模型下載:**
目前所有實驗統一使用 Llama-3-8B (Q4_K_M)

```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## 5. 執行實驗

目前主要的 Active Benchmark 為輸出長度特徵分析（Output Length Characterization）。

```bash
#執行 Benchmark Script 並產生原始 JSON 實驗資料：
python experiments/output_length/output_length.py

#使用以下指令產生統計圖表以及線性迴歸分析：
python experiments/output_length/plot_output_length.py results/profiling/YOUR_RESULT_FILE.json
```