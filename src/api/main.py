from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Edge Vocab Tutir API", lifespan=lifespan)

# API calss and response schemas
class GenerateRequest(BaseModel):
    word: str

class EvalRequest(BaseModel):
    word: str
    definition: str
    user_sentence: str

# API Endpoints

@app.post("/api/generate", response_model=VocabCard)
def generate_card(request: GenerateRequest):
    """Generate VocabCard and store it into DB"""
    # Look up for existence
    existing_data = db.get_word(request.word)

    # Call LLM to generate
    try:
        card = vocab_engine.generate_vocab_card(request.word)
        if not card:
            raise HTTPException(status_code=500, detail="LLM generation failed")

        db.add_word(card.word, card.model_dump())
        return card
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/similar/{word}")
def get_similar(word: str, top_k: int = 5):
    """Look for synonyms in the DB"""
    results = db.get_similar_words(word, top_k)
    if not results:
        raise HTTPException(status_code=404, detail="Word not found in DN or no similar words.")

    return [{"word": w, "similarity": round(s, 4)} for w, s in results]

@app.post("/api/evaluate", response_model=SentenceEvaluation)
def evaluate_sentence(request: EvalRequest):
    """Inspect and correct the user's sentence"""
    eval_result = quiz_engine.evaluate_sentence(
        word=request.word,
        definition=request.definition,
        user_sentence=request.user_sentence
    )
    if not eval_result:
        raise HTTPException(status_code=500, detail="Evaluation failed")

    return eval_result