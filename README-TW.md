# 邊緣運算單字導師 (Edge-Native Vocab Tutor)

本專案旨在具體展現從 [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch) 課程中所學的觀念與架構。這是一個基於邊緣運算、本地推論的語言學習後端系統，專為克服雲端 LLM API 的延遲、成本及結構化輸出限制而設計。

## 目錄

- [引言：問題與解決方案](#引言問題與解決方案)
- [解決的關鍵工程問題 (應用所學)](#解決的關鍵工程問題-應用所學)
- [技術棧與系統架構](#技術棧與系統架構)
- [專案結構](#專案結構)
- [安裝與環境設定](#安裝與環境設定)
- [執行應用程式](#執行應用程式)
- [API 端點總覽](#api-端點總覽)
- [已知限制 (MVP 範圍)](#已知限制-mvp-範圍)

## 引言：問題與解決方案

**1. 背景與挑戰**：
在準備進階英文檢定考試（如 GRE、TOEFL）的過程中，我發現建立個人化的單字庫是一項極度耗時且繁瑣的工作。它需要查詢特定語境下的定義、例句與衍生詞，同時還需留意中英翻譯間的語意偏差。雖然雲端大型語言模型（如 ChatGPT）能提供協助，但卻存在三個致命的缺陷：
*   **成本與依賴性**：完全依賴網際網路連線，且會產生持續性的 API Token 計費成本。
*   **缺乏語意架構**：一般對話式介面缺乏自動化機制（如 Vector Embeddings）來根據語意相似度，動態地將個人單字庫進行分群。
*   **非結構化輸出**：純對話式的 Prompt 互動難以穩定提供程式化學習迴圈（如：自動生成測驗題）所需的嚴謹、結構化資料。

**2. 現況量化**：
若使用傳統軟體（如 Anki 或 Quizlet）手動建立一張包含同義字、衍生句與客製化測驗的高品質單字卡，平均每個單字需耗時約 3 到 5 分鐘。這意味著建立一個基礎的 1000 字 GRE 字庫，將耗費超過 50 小時的純人工作業。雖然市面上有公開的共享字庫，但往往缺乏有效學習所需的個人化特徵。反之，若完全依賴雲端 API 進行自動化，每次生成皆會產生網路延遲（約 2~3 秒）以及長期的計費成本。

**3. 為什麼值得解決（預期影響）**：
為了解決上述痛點，我結合在課程中所學的知識，實作了一套**完全離線運行的 Edge-Native LLM 學習系統**。透過模型量化技術（GGUF, 4-bit），本系統成功將 8B 級別的模型部署於一般消費級筆電上。結合**結構化輸出 (Pydantic)** 與**記憶體內向量資料庫 (Cosine Similarity)**，它保證了單字分群與動態出題的穩定性——同時實現零 API 成本、零網路延遲以及絕對的資料隱私。

## 解決的關鍵工程問題 (應用所學)

本專案將 `ai-engineering-from-scratch` 課程中的理論與架構概念，轉化為實際的工程解決方案：

1. **邊緣推論與模型量化 (Phase 17)**：
   為了克服硬體限制並消除 API 成本，系統採用了 4-bit 量化的 GGUF 模型（`Llama-3-8B-Instruct-Q4_K_M`）。這使得模型能在消費級硬體上進行低延遲的離線推論，並確保 100% 的資料隱私。
2. **結構化輸出與限制解碼 (Phase 11)**：
   雲端 LLM 原生難以保持一致的 JSON 格式。透過將 Pydantic schema 與 `llama-cpp-python` 的限制解碼 (Constrained Decoding) 能力相結合，本系統保證了絕對嚴謹的 JSON 回應，徹底避免後端解析失敗。
3. **語意向量化與單字分群 (Phase 11)**：
   解決了傳統標籤分類中的「詞彙不匹配 (Vocabulary Mismatch)」問題。系統實作了輕量級的記憶體內向量庫（SQLite + `SentenceTransformers`），並透過計算餘弦相似度 (Cosine Similarity) 來動態聚合語意相關的單字。
4. **提示詞工程與輸入消毒 (Phase 11)**：
   為防止使用者輸入造成的定界符衝突 (Delimiter Collision) 與提示詞注入攻擊 (Prompt Injection)，系統採用了 **XML 標籤沙盒 (XML tag sandboxing)**（例如：`<user_sentence>`）。結合精準的角色設定 (Role Prompting) 與負面約束 (Negative Constraints)，成功將模型的行為錨定為客觀的評估者。
5. **思維鏈 (CoT) 批改模組 (Phase 11)**：
   小型語言模型 (SLMs) 在評估句子時，常會產生幻覺式的過度糾錯。評估端點強制導入了 CoT 工作流，迫使模型在給出最終的布林值判定前，必須明確輸出其 `reasoning`（思考過程），從而大幅提升批改的準確度。

## 技術棧與系統架構

* **LLM 引擎**: `llama-cpp-python`
* **模型**: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (4-bit 量化模型)
* **結構化資料驗證**: `Pydantic`
* **語意向量模型 (Embedding)**: `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **向量儲存與檢索**: `SQLite3` + `NumPy` (記憶體內計算餘弦相似度)
* **後端 API**: `FastAPI` + `Uvicorn` (非同步 RESTful API)
* **前端視覺化**: `Streamlit` (仍在開發中)

## 專案結構

```text
edge-vocab-tutor/
├── models/                 # 下載之量化模型存放區
├── src/
│   ├── api/
│   │   └── main.py         # FastAPI 應用程式主程式
│   ├── db/
│   │   └── database.py     # Cosine similarity 檢索邏輯
│   └── llm/
│       ├── generator.py    # 負責生成結構化單字卡
│       └── quiz.py         # CoT 思維鏈造句批改模組
├── requirements.txt        # 專案依賴套件
├── .gitignore
└── README.md
```

## 安裝與環境設定

**1. 系統需求 (Prerequisites)**

* Python 3.10+
* `uv` 套件管理員 (強烈建議)
* Git, wget, build-essential (供 WSL/Ubuntu 環境編譯使用)

**2. 複製專案與安裝依賴套件**

```bash
git clone https://github.com/yourusername/edge-vocab-tutor.git
cd edge-vocab-tutor
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
*(備註：若要在 NVIDIA 裝置上啟用 GPU 加速，請使用 `CMAKE_ARGS="-DGGML_CUDA=on"` 安裝 `llama-cpp-python`)*

**3. 下載量化模型**

```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## 執行應用程式

本系統採用後端 API 與前端儀表板解耦之設計。

**1. 啟動 FastAPI 後端**

啟動 ASGI 伺服器並將模型載入記憶體（作為單例模式載入以防止 OOM 記憶體溢出錯誤）：

```bash
uvicorn src.api.main:app --reload
```
API 說明文件 (Swagger UI) 啟動後將可於以下網址查看：`http://127.0.0.1:8000/docs`

## API 端點總覽

* `POST /api/generate`: 接收目標單字與上下文，生成包含嚴謹型態之 JSON 格式詳細單字卡（含 CEFR 分級、多重字義、同義詞、反義詞）。
* `GET /api/similar/{word}`: 從本地端 SQLite 資料庫中檢索並回傳 5 個語意最相近的單字。
* `POST /api/evaluate`: 接收使用者生成的造句，運行思維鏈 (CoT) 提示，並回傳結構化的批改回饋。

## 已知限制 (MVP 範圍)

* **模型能力限制**：受限於 8B 參數之量化模型，在句子評估時，面對過於複雜的語言邊緣案例，偶爾仍會發生過度糾錯 (Over-correction) 的情況。
* **向量儲存擴展性**：目前的語意向量相似度皆是透過 NumPy 於記憶體內動態計算。此作法對於單字集 (< 10,000 筆) 的效能極佳，但若未來要擴展至百萬級別的企業檢索規模，則需升級為基於 HNSW 架構的向量資料庫（如 Qdrant 或 Milvus）。