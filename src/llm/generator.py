import json
from llama_cpp import Llama
from pydantic import BaseModel, Field
from typing import List

from src.utils.security import sanitize_input

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

        response = self.llm.create_chat_completion(
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
            stop=["<|eot_id|>"]
        )

        result_str = response["choices"][0]["message"]["content"]

        try:
            return VocabCard.model_validate_json(result_str)
        except Exception as e:
            print(f"[System] JSON Validation failed: {e}")
            return None

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