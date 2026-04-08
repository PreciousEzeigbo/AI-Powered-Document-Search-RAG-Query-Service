"""Input sanitization and prompt injection defense utilities."""

import re
from typing import List

from app.schemas import ConversationTurn

# ---------------------------------------------------------------------------
# Control‑character stripping
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def strip_control_chars(text: str) -> str:
    """Remove non-printable control characters (preserves \\n, \\r, \\t)."""
    return _CONTROL_CHAR_RE.sub("", text)


# ---------------------------------------------------------------------------
# Prompt‑injection heuristics
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"pretend\s+you\s+are\s+",
        r"act\s+as\s+(a|an|if)\s+",
        r"new\s+instructions?\s*:",
        r"system\s*prompt\s*:",
        r"override\s+(system|safety|instructions?)",
        r"reveal\s+(your|the)\s+(system|original|initial)\s+(prompt|instructions?)",
        r"output\s+(your|the)\s+(system|original|initial)\s+(prompt|instructions?)",
        r"repeat\s+(your|the)\s+(system|original|initial)\s+(prompt|instructions?)",
        r"\[system\]",
        r"<\s*system\s*>",
        r"###\s*(system|instruction)",
    ]
]


def detect_prompt_injection(text: str) -> bool:
    """Return *True* if *text* matches known prompt‑injection heuristics."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# User input sanitization
# ---------------------------------------------------------------------------

MAX_QUESTION_LENGTH = 2000
MAX_HISTORY_CONTENT_LENGTH = 2000
MAX_HISTORY_TURNS = 10


def sanitize_user_input(text: str, max_length: int = MAX_QUESTION_LENGTH) -> str:
    """Strip control chars and enforce length limit on user input."""
    cleaned = strip_control_chars(text).strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def sanitize_history(history: List[ConversationTurn]) -> List[ConversationTurn]:
    """Sanitize and truncate conversation history.

    Returns a *new* list; the originals are not mutated.
    """
    sanitized: List[ConversationTurn] = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        clean_content = sanitize_user_input(
            turn.content, max_length=MAX_HISTORY_CONTENT_LENGTH
        )
        if clean_content:
            sanitized.append(ConversationTurn(role=turn.role, content=clean_content))
    return sanitized
