"""
Invisible Text Detector Guard
Detects and flags invisible/hidden Unicode characters (e.g., zero-width spaces, 
format control characters, private use area characters).
"""

from typing import List, Dict, Any
import unicodedata
import logging

logger = logging.getLogger("guardrails")

# Unicode categories to flag:
# Cf: Other, format (e.g. Zero Width Joiner)
# Co: Other, private use
# Cn: Other, not assigned
BANNED_CATEGORIES = {"Cf", "Co", "Cn"}

def detect_invisible_text(text: str) -> List[Dict[str, Any]]:
    """
    Scans for invisible or non-printable Unicode characters.
    Returns a list of issues if any are found.
    """
    if not text:
        return []

    issues = []
    found_chars = []
    
    # Quick check for non-ASCII
    if not any(ord(char) > 127 for char in text):
        return []

    for char in text:
        cp = ord(char)
        category = unicodedata.category(char)
        name = unicodedata.name(char, "Unknown")
        
        is_invisible = False
        
        # 1. Standard non-printable categories
        if category in BANNED_CATEGORIES:
            is_invisible = True
            
        # 2. Variation Selectors (often categorized as 'Mn')
        # VS1-VS16: U+FE00-U+FE0F
        # VS17-VS256: U+E0100-U+E01EF
        elif (0xFE00 <= cp <= 0xFE0F) or (0xE0100 <= cp <= 0xE01EF):
            is_invisible = True
            
        # 3. Specifically look for "VARIATION SELECTOR" in the name
        elif "VARIATION SELECTOR" in name:
            is_invisible = True
            
        # 4. Non-standard whitespace (category 'Zs' except regular space)
        # Standard spaces are U+0020 (Space). Newlines/tabs are handled by category logic or allowed.
        elif category == "Zs" and cp != 0x0020:
            is_invisible = True
            
        # 5. Other separators (Zl: Line, Zp: Paragraph)
        elif category in {"Zl", "Zp"}:
            is_invisible = True

        if is_invisible:
            found_chars.append({
                "char": char,
                "hex": hex(cp),
                "name": name,
                "category": category
            })

    if found_chars:
        # Create a sample of detected hex codes for the description
        hex_samples = ", ".join(set(c["hex"] for c in found_chars[:5]))
        if len(set(c["hex"] for c in found_chars)) > 5:
            hex_samples += "..."
            
        logger.warning(f"Invisible characters detected: {hex_samples}")
        
        issues.append({
            "guard": "invisible_text_detector",
            "description": f"Found {len(found_chars)} invisible or non-standard Unicode characters ({hex_samples})",
            "matched": hex_samples,
            "score": 1.0,
            "details": found_chars
        })

    return issues
