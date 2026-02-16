import json
import re
import logging
from typing import List, Dict, Any

from ..core.messages import LogMessages

logger = logging.getLogger("lexisight")

def parse_json_output(output_text: str) -> List[Dict[str, Any]]:
    """
    Parses a JSON string (potentially wrapped in markdown code blocks) 
    into a list of dictionaries.
    """
    blocks = []
    try:
        # Clean json string from potential markdown code blocks
        json_str = output_text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        
        blocks = json.loads(json_str.strip())
    except Exception as e:
        logger.warning(LogMessages.JSON_PARSE_FAIL.format(e))
        # Fallback: try to find list in string
        try:
            match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if match:
                blocks = json.loads(match.group())
        except Exception as fallback_error:
            logger.error(LogMessages.JSON_FALLBACK_FAIL.format(fallback_error))
            
    return blocks
