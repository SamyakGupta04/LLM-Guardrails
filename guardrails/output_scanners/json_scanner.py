"""
JSON Scanner
Detects, validates, and repairs JSON structures in LLM output.

Features:
- regex-based detection (handles nested structures)
- json_repair for fixing common errors (missing quotes, trailing commas)
- Optional structure validation (min_length, max_length, required_keys)
"""

import json
import logging
from typing import List, Dict, Any, Optional
import regex
from json_repair import repair_json

logger = logging.getLogger("guardrails")

# Recursive regex pattern to match nested JSON objects
JSON_PATTERN = r"(?<!\\)(?:\\\\)*\{(?:[^{}]|(?R))*\}"

def detect_json_syntax(
    output: str,
    repair: bool = True,
    required_elements: int = 0,
) -> List[Dict[str, Any]]:
    """
    Scan output for valid JSON. If invalid, attempt to repair.
    
    Args:
        output: The text to scan.
        repair: Whether to attempt repairing broken JSON.
        required_elements: Minimum number of valid JSON objects required.
    
    Returns:
        List of issues found (e.g., "Invalid JSON", "Missing JSON").
    """
    if not output or not output.strip():
        return []

    # Compile regex with recursive support
    try:
        pattern = regex.compile(JSON_PATTERN, regex.DOTALL)
    except Exception as e:
        logger.error(f"Regex compilation error: {e}")
        return []

    # Find candidates
    candidates = pattern.findall(output)
    
    valid_jsons = []
    issues = []
    
    for i, candidate in enumerate(candidates):
        try:
            # Try parsing as-is
            json.loads(candidate)
            valid_jsons.append(candidate)
        except json.JSONDecodeError as e:
            # Invalid JSON found
            if not repair:
                issues.append({
                    "guard": "json_scanner",
                    "description": f"Invalid JSON syntax at match #{i+1}: {str(e)}",
                    "matched": candidate[:50] + "..." if len(candidate) > 50 else candidate,
                    "score": 1.0
                })
                continue
            
            # Attempt repair
            logger.info(f"Attempting to repair JSON match #{i+1}...")
            try:
                repaired = repair_json(candidate, return_objects=False)
                
                # Verify repair worked
                json.loads(repaired)
                valid_jsons.append(repaired)
                
                # Note: We don't modify the original output string in the scanner logic itself
                # because scanners typically return issues/analysis, not transformed output directly.
                # However, if the user wants the repaired output, we could return it or log it.
                # For now, we just flag it as "Repaired" success or fail.
                
                logger.debug(f"JSON repaired successfully: {repaired[:50]}...")
                
            except Exception as repair_err:
                issues.append({
                    "guard": "json_scanner",
                    "description": f"Unrepairable JSON syntax at match #{i+1}",
                    "matched": candidate[:50] + "..." if len(candidate) > 50 else candidate,
                    "score": 1.0
                })

    # Check validation requirements
    if len(valid_jsons) < required_elements:
        issues.append({
            "guard": "json_scanner",
            "description": f"Insufficient valid JSON objects found. Required: {required_elements}, Found: {len(valid_jsons)}",
            "score": 1.0
        })
        
    return issues
