import json
import time
from llama_cpp import Llama
from pydantic import BaseModel, Field
from typing import List

from src.utils.security import sanitize_input
from src.utils.parser import extract_json_string

# define a singal meaning of a word
class WordSense(BaseModel):
    part_of_speech: str = Field(..., description="Part of speech (e.g., noun, verb, adjective)")
    definition: str = Field(..., description="Definition of the word")

# vocab card model
class VocabCard(BaseModel):
    word: str = Field(..., description="The targeted English word")
    cefr_level: str = Field(..., description="Estimated CEFR level (e.g. B2, C1, C2)")
    senses: List[WordSense] = Field(..., max_length=3, description="List of different meanings/senses of the word")
    synonyms: List[str] = Field(..., max_length=3, description="3 synonyms")
    antonyms: List[str] = Field(..., max_length=3, description="3 antonyms")
    example_sentence: str = Field(..., description="A GRE-level example sentence")

class LLMEngine:
    def __init__(self, llm: Llama):
        self.llm = llm
        self.schema = VocabCard.model_json_schema()

    def generate_vocab_card(self, word: str) -> VocabCard:
        safe_word = sanitize_input(word)

        prompt = (
            f"Analyze the English word provided below.\n"
            f"Provide a vocabulary card. Keep definitions and examples extremely concise (under 15 words).\n\n"
            f"<word>\n{safe_word}\n</word>"
        )

        response_stream = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a precise dictionary API. Output strict, concise JSON without any preamble."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_object",
                "schema": self.schema,
            },
            temperature=0.3,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_tokens=-1,
            stop=["<|eot_id|>"],
            stream=True
        )

        result_str = ""
        chunk_count = 0
        t_first_content = None
        t_last_content = None

        t_llm_start = time.perf_counter()

        for chunk in response_stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    current_time = time.perf_counter()

                    if t_first_content is None:
                        t_first_content = current_time

                    t_last_content = current_time
                    result_str += content
                    chunk_count += 1

        # more detailed derived metrics
        if t_first_content is not None:
            true_output_tokens = len(self.llm.tokenize(result_str.encode('utf-8')))

            ttft = t_first_content - t_llm_start
            decode_time = t_last_content - t_first_content

            tpot = decode_time / (chunk_count - 1) if chunk_count > 1 else None
            tpot_display = f"{tpot*1000:.2f}ms/token" if tpot is not None else "N/A"

            print(f"[Profiler - LLM Tier] "
                  f"TTFT: {ttft*1000:.2f}ms | "
                  f"Decode: {decode_time:.4f}s | "
                  f"Chunks: {chunk_count} | "
                  f"Tokens: {true_output_tokens} | "
                  f"TPOT: {tpot_display}")
        else:
            print("[Profiler - LLM Tier] No content generated.")

# For testing
if __name__ == "__main__":
    MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

    print("[System] Initializing LLM Engine...")
    engine = LLMEngine(MODEL_PATH)

    target_word = "mitigate"
    print(f"[System] Generating card for: {target_word}...")

    card = engine.generate_vocab_card(target_word)
    print("\n--- Output Result ---")
    print(json.dumps(card.model_dump(), indent=2, ensure_ascii=False))