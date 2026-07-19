import re
from typing import Optional

def sanitize_input(text: Optional[str]) -> Optional[str]:
    """Filter user's input to prevent prompt injection and XML conficts"""
    if not text:
        return text

    sanitized = re.sub(r'[<>]', '', text)
    return sanitized.strip()