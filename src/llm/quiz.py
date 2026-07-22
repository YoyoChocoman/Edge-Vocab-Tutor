import json
from llama_cpp import Llama
from pydantic import BaseModel, Field

from src.utils.security import sanitize_input
from src.utils.parser import extract_json_string

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
        safe_word = sanitize_input(word)
        safe_def = sanitize_input(definition)
        safe_sentence = sanitize_input(user_sentence)

        # Prompt Engineering: use XML to isolate all labels (now discarded)
        prompt = (
            "Evaluate if the user's sentence uses the target word correctly in terms of semantics, collocation, and grammar.\n"
            "Think step-by-step in the 'reasoning' field before providing the final boolean judgment and feedback.\n\n"
            f"<target_word>\n{safe_word}\n</target_word>\n\n"
            f"<target_definition>\n{safe_def}\n</target_definition>\n\n"
            f"<user_sentence>\n{safe_sentence}\n</user_sentence>"
        )

        print(f"[System] Sending sentence to LLM for evaluation...")

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are extremely strict GRE-level English linguistic judge. You strictly evaluate vocabulary usage based on definition, part of speech, and logic. Especially the part of speech. Output strictly in the requested JSON format."},

                # Case 1：Part Of Speech Mistaken (False Negative Defense)
                {"role": "user", "content": "<target_word>\nmitigate\n</target_word>\n<target_definition>\nMake a situation less severe\n</target_definition>\n<user_sentence>\nHe felt very mitigate after passing the exam.\n</user_sentence>"},
                {"role": "assistant", "content": '{"reasoning": "The word \'mitigate\' is a verb, but it is incorrectly used here as an adjective. This is a fatal grammatical error.", "is_appropriate": false, "feedback": "You used \'mitigate\' as an adjective, but it is a verb. A correct sentence would be: \'Nothing could mitigate his anxiety.\'"}'},

                # Case 2：Logic Paradox (False Negative Defense)
                {"role": "user", "content": "<target_word>\nalleviate\n</target_word>\n<target_definition>\nMake a problem less severe\n</target_definition>\n<user_sentence>\nDrinking large amounts of seawater will alleviate your thirst when you are lost at sea.\n</user_sentence>"},
                {"role": "assistant", "content": '{"reasoning": "Grammatically correct, but logically contradictory. Seawater exacerbates thirst rather than alleviating it, violating basic semantic logic.", "is_appropriate": false, "feedback": "Your sentence contains a logical contradiction. Seawater does not alleviate thirst. Consider replacing \'seawater\' with \'fresh water\'."}'},

                # True Task
                {"role": "user", "content": f"<target_word>\n{safe_word}\n</target_word>\n<target_definition>\n{safe_def}\n</target_definition>\n<user_sentence>\n{safe_sentence}\n</user_sentence>"}
            ],
            response_format={
                "type": "json_object",
                "schema": self.schema,
            },
            temperature=0.0,
            max_tokens=-1,
            stop=["<|eot_id|>"]
        )

        result_str = response["choices"][0]["message"]["content"]
        clean_json = extract_json_string(result_str)
        parsed_dict = json.loads(clean_json, strict=False)

        try:
            return SentenceEvaluation.model_validate(parsed_dict)
        except Exception as e:
            print(f"[System] Evaluation parsing failed: {e}")
            print(f"[System] Raw output: {result_str}")
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