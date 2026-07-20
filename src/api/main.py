import asyncio
import gc
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from contextlib import asynccontextmanager
from llama_cpp import Llama

from src.llm.generator import LLMEngine, VocabCard
from src.llm.quiz import QuizEngine, SentenceEvaluation
from src.db.database import VocabDatabase

db = None
vocab_engine = None
quiz_engine = None
llm_instance = None

llm_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, vocab_engine, quiz_engine, llm_instance
    print("[Server] Status: Activating (loading database and LLM model)")

    db = VocabDatabase("vocab.db")

    MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    llm_instance = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False
    )

    vocab_engine = LLMEngine(llm_instance)
    quiz_engine = QuizEngine(llm_instance)

    print("[Server] Status: Ready")

    # Hand the control to FastAPI
    yield

    print("[Server] Status: Shutting Down (Release GPU resources and DB connections)")

    del vocab_engine
    del quiz_engine
    del llm_instance

    if db and db.conn:
        db.conn.close()

    gc.collect()
    print("[Server] Status: Done")

app = FastAPI(title="Edge Vocab Tutir API", lifespan=lifespan)

# API class and response schemas
class GenerateRequest(BaseModel):
    word: str

class EvalRequest(BaseModel):
    word: str
    definition: str
    user_sentence: str

# API Endpoints

@app.post("/api/generate", response_model=VocabCard)
async def generate_card(request: GenerateRequest):
    """Generate VocabCard and store it into DB (with Mutex Lock)"""
    # Step 1: if the word exists in DB, return it directly
    request.word = request.word.lower().strip()
    existing_data = db.get_word(request.word)
    if existing_data:
        print(f"[API] Cache Hit: '{request.word}'")
        return existing_data

    # Step 2: otherwise, generate LLM logic
    async with llm_lock:
        try:
            print(f"[API] Cache Miss: '{request.word}', generating...")
            card = await asyncio.to_thread(vocab_engine.generate_vocab_card, request.word)

            if not card:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM failed to generate or parse valid JSON.")

            db.add_word(card.word, card.model_dump())
            return card
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/api/similar/{word}")
async def get_similar(word: str, top_k: int = 5):
    """Look for synonyms in the DB (No Lock)"""
    # Case 1: the word doesn't exist
    word = word.lower().strip()
    existing_data = db.get_word(word)
    if not existing_data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Word '{word}' not found in database.")

    results = db.get_similar_words(word, top_k)

    # Case 2: the word has no similarity
    if not results:
        return []

    return [{"word": w, "similarity": round(s, 4)} for w, s in results]

@app.post("/api/evaluate", response_model=SentenceEvaluation)
async def evaluate_sentence(request: EvalRequest):
    """Inspect and correct the user's sentence (with Mutex Lock)"""
    async with llm_lock:
        try:
            eval_result = await asyncio.to_thread(
                quiz_engine.evaluate_sentence,
                request.word,
                request.definition,
                request.user_sentence
            )

            if not eval_result:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM evaluation failed or returned invalid JSON.")

            return eval_result
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))