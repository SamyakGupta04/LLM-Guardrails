"""
PII Detection Guard
Regex-based approach for sensitive PII detection (Indian context).
Includes preprocessing to handle obfuscation and encoding.
"""

from typing import List, Dict, Any, Tuple

import re
import unicodedata
import base64
import logging

logger = logging.getLogger("guardrails")


# =============================================================================
# Verhoeff Algorithm for Aadhaar Checksum Validation
# =============================================================================

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _validate_aadhaar_checksum(aadhaar: str) -> bool:
    """Validate Aadhaar using Verhoeff algorithm."""
    digits = [int(d) for d in aadhaar if d.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_text(text: str) -> Tuple[str, List[str]]:
    """
    Full preprocessing pipeline for PII detection.
    Returns (preprocessed_text, list_of_transformations_applied).
    """
    transformations = []
    processed = text
    
    # 1. Remove zero-width and format characters
    clean = ''.join(c for c in processed if unicodedata.category(c) != 'Cf')
    if clean != processed:
        transformations.append("removed_format_chars")
        processed = clean
    
    # 2. Unicode normalization
    clean = unicodedata.normalize('NFKD', processed)
    if clean != processed:
        transformations.append("normalized_unicode")
        processed = clean
    
    # 3. Decode Base64 segments (with limits)
    clean, was_decoded = _decode_base64(processed)
    if was_decoded:
        transformations.append("decoded_base64")
        processed = clean
    
    # 4. Collapse spaced characters (e.g., "A B C D E 1 2 3 4 F" -> "ABCDE1234F")
    # Updated to include alphanumeric characters, not just letters
    clean = re.sub(r'(?:\b[a-zA-Z0-9]\s+){3,}[a-zA-Z0-9]\b', 
                   lambda m: m.group(0).replace(' ', ''), processed)
    if clean != processed:
        transformations.append("collapsed_spaced_chars")
        processed = clean
    
    # 5. Normalize common obfuscations (email patterns)
    subs = [
        (r'\s*[\[\(]\s*at\s*[\]\)]\s*', '@'),
        (r'\s*[\[\(]\s*dot\s*[\]\)]\s*', '.'),
        (r'\s+at\s+', '@'),
        (r'\s+dot\s+', '.'),
    ]
    for pattern, repl in subs:
        new_text = re.sub(pattern, repl, processed, flags=re.IGNORECASE)
        if new_text != processed:
            transformations.append("normalized_substitutions")
            processed = new_text
            # Don't break - apply all substitutions
    
    # 6. Collapse whitespace
    processed = ' '.join(processed.split())
    
    return processed, transformations


def _decode_base64(text: str, max_segments: int = 3) -> Tuple[str, bool]:
    """Decode Base64 segments with safety limits."""
    pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
    decoded_any = False
    count = 0
    
    def try_decode(match):
        nonlocal decoded_any, count
        if count >= max_segments:
            return match.group(0)
        segment = match.group(0)
        if len(segment) > 2048 or len(segment) % 4 != 0:
            return segment
        try:
            decoded = base64.b64decode(segment).decode('utf-8', errors='ignore')
            if decoded.isprintable() and len(decoded) > 5:
                decoded_any = True
                count += 1
                return f" {decoded} "
        except Exception:
            pass
        return segment
    
    return pattern.sub(try_decode, text), decoded_any


# =============================================================================
# PII Labels and Regex Patterns
# =============================================================================

# Pre-compiled regex patterns for Indian PII
_UPI_HANDLES = (
    "upi|paytm|okaxis|okicici|okhdfcbank|ybl|ibl|axl|sbi|apl|rapl|icici|"
    "hdfcbank|axisbank|kotak|indus|federal|rbl|idbi|pnb|boi|cbi|union|bob|"
    "canara|syndicate|allahabad|andhra|vijaya|dena|uco|obc|iob|mahabank|"
    "centralbank|fam|jupitermoney|slice|fi|niyobank|googlepe|phonepe|amazonpay"
)

COMPILED_PATTERNS = {
    # Indian Passport: 1 uppercase letter + 7 digits (case-insensitive)
    "passport": re.compile(r"\b[A-Z][0-9]{7}\b", re.IGNORECASE),
    # UPI ID: alphanumeric@handle
    "upi": re.compile(rf"\b[a-zA-Z0-9._-]+@(?:{_UPI_HANDLES})\b", re.IGNORECASE),
    # Indian Phone: 10 digits starting with 6-9
    "phone_india": re.compile(r"(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}\b"),
    # Aadhaar: 12 digits, cannot start with 0 or 1
    "aadhaar": re.compile(r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    # PAN: Strict format - 4th char must be [ABCFGHLJPTK] (case-insensitive)
    "pan": re.compile(r"\b[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]\b", re.IGNORECASE),
    # Email: standard format, exclude UPI handles
    "email": re.compile(rf"\b[a-zA-Z0-9._%+-]+@(?!{_UPI_HANDLES})[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}}\b", re.IGNORECASE),
}


# =============================================================================
# Main Detection Function
# =============================================================================

def detect_pii(text: str) -> List[Dict[str, Any]]:
    """
    Detect PII using Regex patterns.
    Applies preprocessing to handle obfuscation.
    """
    if not text or not text.strip():
        return []
    
    issues = []
    seen_values = set()
    
    # Preprocess
    processed, transformations = preprocess_text(text)
    if transformations:
        logger.debug(f"PII preprocessing: {transformations}")
    
    # Regex Detection with validation
    for name, pattern in COMPILED_PATTERNS.items():
        for match in pattern.finditer(processed):
            value = match.group(0)
            
            # Extra validation for Aadhaar (Verhoeff checksum)
            if name == "aadhaar":
                if not _validate_aadhaar_checksum(value):
                    logger.debug(f"Aadhaar failed checksum: {value[:4]}...")
                    continue
            
            masked = _mask_value(value)
            if masked not in seen_values:
                seen_values.add(masked)
                issues.append({
                    "guard": "pii_detector_regex",
                    "pii_type": name,
                    "description": f"Detected {name} (Regex Pattern)",
                    "matched": masked,
                    "score": 1.0
                })

    return issues


def _mask_value(value: str) -> str:
    """Mask a PII value, showing only first 2 and last 2 characters."""
    clean = re.sub(r'[\s\-]', '', value)
    if len(clean) > 4:
        return clean[:2] + "*" * (len(clean) - 4) + clean[-2:]
    return "****"
