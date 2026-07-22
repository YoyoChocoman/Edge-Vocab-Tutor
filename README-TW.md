# 邊緣運算單字導師

本專案旨在具體展現從 [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) 課程中所學的觀念與系統架構。這是一個基於邊緣運算、本地推論的語言學習後端系統，專為克服雲端 LLM API 的延遲、成本、結構化輸出限制以及資料隱私風險而設計。

## 📑 目錄

- [引言：問題與解決方案](#引言問題與解決方案)
- [關鍵架構實作](#關鍵架構實作)
- [技術棧與系統架構](#技術棧與系統架構)
- [專案結構](#專案結構)
- [安裝與環境設定](#安裝與環境設定)
- [執行應用程式](#執行應用程式)
- [系統評估與工程妥協](EVALUATION-TW.md) *(查看詳細模型指標與評估報告)*

## 引言：問題與解決方案

**1. 背景與挑戰**
在準備進階英文檢定考試（如 GRE、TOEFL）的過程中，建立個人化的單字庫是一項極度耗時且繁瑣的工作。雖然雲端大型語言模型（如 ChatGPT）能提供協助，但存在三大工程與隱私缺陷：
*   **成本與隱私**：完全依賴網際網路連線，產生持續性的 API Token 計費成本，且使用者的學習數據暴露於雲端。
*   **缺乏語意架構**：傳統對話式介面無法將使用者的個人單字庫進行語意向量化（Embedding），難以實現動態的相似詞分群。
*   **非結構化輸出**：純對話式的 Prompt 難以穩定提供程式化學習迴圈所需的嚴謹 JSON 格式。

**2. 預期影響 (解決方案)**
為解決上述痛點，本專案實作了一套**完全離線運行的 Edge-Native LLM 學習系統**。透過 4-bit 模型量化技術（GGUF），成功將 8B 級別的模型部署於一般消費級硬體（VRAM 佔用僅約 5GB）。結合**結構化輸出 (Pydantic)** 與**記憶體內向量資料庫 (Cosine Similarity)**，系統保證了單字分群與動態出題的絕對穩定性——實現零 API 成本、零網路依賴以及完全的資料隱私。

## 關鍵架構實作

1. **邊緣推論與模型量化 (Phase 17)**：
   部署 `Llama-3-8B-Instruct-Q4_K_M` 於消費級硬體運行。4-bit 量化技術有效壓縮了記憶體佔用（Memory Footprint），同時維持了模型所需的推論邏輯能力。
2. **強健解析與限制解碼 (Phase 11)**：
   初期壓力測試顯示，即使導入 Pydantic，8B 模型在推論時仍有 23% 的機率因生成 Markdown 標記導致 JSON 解析崩潰。本系統實作了 Regex 強制提取器與寬鬆解析機制 (`strict=False`)，成功將 API 的 **JSON 解析成功率提升並穩定至 100%**。
3. **語意向量化與單字分群 (Phase 11)**：
   實作輕量級的記憶體內向量庫（SQLite + `SentenceTransformers`），透過動態計算餘弦相似度 (Cosine Similarity) 解決「詞彙不匹配 (Vocabulary Mismatch)」問題，自動聚合語意相關的單字，免除架設重型向量資料庫的開銷。
4. **輸入消毒與 XML 沙盒 (Phase 11)**：
   摒棄傳統引號定界符，改用 **XML 標籤沙盒 (XML tag sandboxing)**（例如 `<user_sentence>`）隔離使用者輸入，有效防止定界符衝突並抵禦提示詞注入攻擊 (Prompt Injection)。
5. **多輪少樣本思維鏈批改 (Phase 11)**：
   為壓制小型模型 (SLM) 常見的「幻覺性過度糾錯」，將評估模組的提示詞從零樣本 (Zero-Shot) 升級為 **多輪少樣本思維鏈 (Multi-Turn Few-Shot CoT)**。有關誤判率 (False Positive/Negative) 大幅降低的細節，請參閱 [評估報告](EVALUATION.md)。

## 技術棧與系統架構

* **LLM 推論引擎**: `llama-cpp-python` (C++ 後端最佳化)
* **語言模型**: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (4-bit 量化)
* **結構化資料驗證**: `Pydantic`
* **語意向量模型**: `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **向量儲存與檢索**: `SQLite3` + `NumPy` (記憶體內運算)
* **後端 API**: `FastAPI` + `Uvicorn` (內建 Mutex Lock 控制推論併發)
* **前端視覺化**: `Streamlit`

## 專案結構

```text
edge-vocab-tutor/
├── models/                 # 下載之量化模型存放區
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI 應用程式主程式
│   ├── db/
│   │   └── database.py     # 檢索邏輯與 SQLite 資料表設計
│   └── llm/
│       ├── generator.py    # 結構化單字卡生成模組
│       └── quiz.py         # Few-shot CoT 批改模組
├── tests/
│   ├── eval_dataset.json   # 黃金測試資料集 (100筆 Edge cases)
│   ├── run_eval.py         # 自動化 FP/FN 計算與報表生成腳本
│   └── test_load.py        # 端到端 API 壓力測試腳本
├── requirements.txt
├── .gitignore
├── EVALUATION.md           # 系統評估與工程權衡分析報告
└── README.md
```

## 安裝與環境設定

**1. 系統需求**
* Python 3.10+
* `uv` 套件管理員 (建議使用)
* Git, wget, build-essential (供 WSL/Ubuntu 環境編譯使用)

**2. 複製專案與安裝依賴套件**
```bash
git clone https://github.com/yourusername/edge-vocab-tutor.git
cd edge-vocab-tutor
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
*(備註：若需在 NVIDIA 裝置啟用 GPU 加速，請於安裝 `llama-cpp-python` 時加上環境變數 `CMAKE_ARGS="-DGGML_CUDA=on"`)*

**3. 下載量化模型**
```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## 執行應用程式

**1. 啟動 FastAPI 後端**
```bash
uvicorn src.api.main:app --reload
```
*API 說明文件 (Swagger UI) 啟動後將位於：`http://127.0.0.1:8000/docs`*

**2. 啟動 Streamlit 前端 (請於獨立的終端機執行)**
```bash
source .venv/bin/activate
streamlit run src/ui/app.py
```