"""
Code Detection Guard (Heuristic)
Detects and blocks code submissions using Regex patterns and density heuristics.
Lightweight, fast, and does not require ML models.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("guardrails")

# =============================================================================
# Configuration
# =============================================================================

CODE_THRESHOLD = 0.6  # Heuristic score threshold (0.0 to 1.0)

# =============================================================================
# Regular Expressions
# =============================================================================

# Strong signals: These almost always indicate code
STRONG_PATTERNS = [
    # Function definitions
    r"def\s+[a-zA-Z_]\w*\s*\(",             # Python: def foo(
    r"function\s+[a-zA-Z_]\w*\s*\(",        # JS: function foo(
    r"[a-zA-Z_]\w*\s*=\s*\([^)]*\)\s*=>",   # JS arrow: const foo = () =>
    r"public\s+class\s+[a-zA-Z_]",          # Java/C#: public class Foo
    r"class\s+[a-zA-Z_]\w*\s*\{",           # Generic class
    
    # Imports
    r"^import\s+[a-zA-Z0-9_.]+\s*$",        # Python: import os
    r"^from\s+[a-zA-Z0-9_.]+\s+import",     # Python: from x import y
    r"#include\s+<[a-zA-Z0-9_.]+>",         # C/C++: #include <stdio.h>
    
    # Control flow with braces
    r"if\s*\([^)]+\)\s*\{",                 # C-style if: if (condition) {
    r"for\s*\([^)]+\)\s*\{",                # C-style for
    r"while\s*\([^)]+\)\s*\{",              # C-style while
    
    # SQL
    r"SELECT\s+.+\s+FROM\s+.+",             # SQL Select
    r"INSERT\s+INTO\s+.+\s+VALUES",         # SQL Insert
    
    # Shell/System
    r"rm\s+-rf\s+",                         # rm -rf
    r"sudo\s+[a-zA-Z]",                     # sudo cmd
    r"wget\s+http",                         # wget
    r"curl\s+-",                            # curl flags
]

# Weak signals: Common in code but also in text
WEAK_PATTERNS = [
    r"[a-zA-Z_]\w*\s*=\s*\[.*\]",           # List assignment: x = [...]
    r"[a-zA-Z_]\w*\s*=\s*\{.*\}",           # Dict/Obj assignment: x = {...}
    r"[a-zA-Z_]\w*\s*=\s*[\"'].*[\"']",     # String assignment: x = "foo"
    r"print\s*\(.*\)",                      # print(...)
    r"console\.log\(",                      # console.log(
    r"return\s+.*;",                        # return x;
    r"var\s+[a-zA-Z_]",                     # var x
    r"const\s+[a-zA-Z_]",                   # const x
    r"let\s+[a-zA-Z_]",                     # let x
    r"\$\([\"'].*[\"']\)",                  # jQuery $(...)
]

# Syntax characters that appear frequently in code
SYNTAX_CHARS = set("{}[]();=<>")

def calculate_density(text: str) -> float:
    """Calculate the density of syntax characters."""
    if not text.strip():
        return 0.0
    
    count = sum(1 for c in text if c in SYNTAX_CHARS)
    return min(1.0, count / len(text) * 10)  # simple scaling

# =============================================================================
# Main Detection Function
# =============================================================================

def detect_code(text: str, threshold: float = CODE_THRESHOLD) -> List[Dict[str, Any]]:
    """
    Detect if the input contains code using heuristic scoring.
    """
    if not text or not text.strip():
        return []

    issues = []
    score = 0.0
    matched_patterns = []

    # 1. Check Strong Patterns (High weight)
    for p in STRONG_PATTERNS:
        if re.search(p, text, re.MULTILINE | re.IGNORECASE):
            score += 0.7
            matched_patterns.append("strong_syntax")
            # Break if we already have a very high score? 
            # No, assume cumulative evidence.

    # 2. Check Weak Patterns (Low weight)
    for p in WEAK_PATTERNS:
        if re.search(p, text, re.MULTILINE):
            score += 0.2
            matched_patterns.append("weak_syntax")

    # 3. Density Check
    density = calculate_density(text)
    if density > 0.3:
        score += 0.2
        matched_patterns.append("high_syntax_density")

    # Normalize score
    score = min(1.0, score)

    if score >= threshold:
        logger.warning(f"Code detected: score={score:.2f}, threshold={threshold}")
        
        preview = text[:100] + "..." if len(text) > 100 else text
        
        issues.append({
            "guard": "code_detector_heuristic",
            "description": f"Code submission detected (confidence: {score:.2f})",
            "matched": preview,
            "score": score,
        })
    else:
        logger.debug(f"No code detected: score={score:.2f}")

    return issues
