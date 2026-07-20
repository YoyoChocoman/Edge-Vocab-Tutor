import re

def extract_json_string(raw_text: str) -> str:
    """Extract JSON blocks from the model's original string using regex."""
    if not raw_text:
        return ""

    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        return match.group(0)

    return raw_text