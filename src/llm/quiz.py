import json
from llama_cpp import Llama
from pydantic import BaseModel, Field

# define evaluation Schema to force LLM into reasoning
class SentenceEvaluation(BaseModel):
    reasoning: str = Field(..., description="Step-by-step analysis of the user's sentence regarding grammar, semantics, and GRE-level appropriateness.")
    is_appropriate: bool = Field(..., description="True if the word is used correctly, False otherwise.")
    feedback: str = Field(..., description="Actionable feedback or correction for the user.")

class QuizEngine:
    def __init__(self, llm: Llama):
        self.llm = llm
        self.schema = SentenceEvaluation.model_json_schema()

    def evaluate_sentence(self, word: str, definition: str, user_sentence: str) -> SentenceEvaluation:
        prompt = (
            f"Word: '{word}'\n"
            f"Target Definition: {definition}\n"
            f"User's Sentence: \"{user_sentence}\"\n\n"
            "Evaluate if the user's sentence uses the word correctly in terms of semantics, collocation, and grammar. "
            "Think step-by-step in the 'reasoning' field before providing the final boolean judgment and feedback."
        )

        print(f"[System] Sending sentence to LLM for evaluation...")

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a strict and professional GRE linguistic tutor. You demand precise vocabulary usage."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_object",
                "schema": self.schema,
            },
            temperature=0.1,
            max_tokens=-1,
            stop=["<|eot_id|>"]
        )

        result_str = response["choices"][0]["message"]["content"]

        try:
            return SentenceEvaluation.model_validate_json(result_str)
        except Exception as e:
            print(f"[System] Evaluation parsing failed: {e}\nRaw output: {result_str}")
            return None

# For Test
if __name__ == "__main__":
    MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

    print("[System] Initializing Quiz Engine...")
    engine = QuizEngine(MODEL_PATH)

    word = "mitigate"
    definition = "Make a situation less severe, harmful, or painful"

    # Case 1：Correct
    good_sentence = "The government implemented new flood defenses to mitigate the damage caused by heavy rains."
    print("\n--- Test Case 1: Correct Usage ---")
    print(f"User Sentence: {good_sentence}")
    eval_good = engine.evaluate_sentence(word, definition, good_sentence)
    if eval_good:
        print(json.dumps(eval_good.model_dump(), indent=2, ensure_ascii=False))

    # Case 2：Part of speech misused
    bad_sentence = "He was very mitigate when he heard the bad news."
    print("\n--- Test Case 2: Incorrect Usage ---")
    print(f"User Sentence: {bad_sentence}")
    eval_bad = engine.evaluate_sentence(word, definition, bad_sentence)
    if eval_bad:
        print(json.dumps(eval_bad.model_dump(), indent=2, ensure_ascii=False))