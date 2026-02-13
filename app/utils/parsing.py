import json
import re
import logging
from typing import List, Dict, Any

from ..core.messages import LogMessages

logger = logging.getLogger("lightonocr")

def parse_text_output(output_text: str) -> str:
    """
    Clean up raw model output text.
    Strips any leading/trailing whitespace and markdown code fences if present.
    """
    text = output_text.strip()
    if text.startswith("```"):
        # Remove markdown code block wrapper
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()
