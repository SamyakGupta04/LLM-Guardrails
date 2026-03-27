"""
Enhanced Sensitive Data Scanner (Deanonymization)
Detects PII in LLM outputs using hybrid approach:
1. Regex patterns (fast, precise for structured IDs)
2. Lightweight ONNX NER (catches names, addresses, etc. ~80MB)
3. False positive filtering (blocklist, context, checksum)

Focused on reducing false positives while maintaining high recall.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger("guardrails")

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_THRESHOLD = 0.7  # Minimum confidence for NER entities
# Using a quantified ONNX model for speed and size (~80MB)
MODEL_NAME = "dslim/bert-base-NER" 
USE_ONNX = True # Flag to use ONNX runtime if available (much smaller/faster)

# Model cache directory
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_LLMGUARD_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
MODEL_CACHE_DIR = os.path.join(_LLMGUARD_DIR, "models")

# =============================================================================
# False Positive Blocklist
# =============================================================================

# Common words that NER incorrectly flags as entities
_FP_BLOCKLIST = {
    # Generic terms
    "india", "indian", "account", "bank", "customer", "user", "client",
    "amount", "balance", "transaction", "payment", "transfer", "credit",
    "debit", "loan", "interest", "rate", "percent", "rupee", "rupees",
    "inr", "usd", "eur", "gbp", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february",
    "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "today", "tomorrow", "yesterday",
    # Banking terms
    "neft", "rtgs", "imps", "upi", "ifsc", "swift", "iban", "bic",
    "emi", "roi", "kyc", "cibil", "nach", "ecs", "mandate", "nominee",
    # Common abbreviations
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "ltd", "pvt", "inc",
    # IndusInd specific
    "indusind", "indus", "axis", "hdfc", "icici", "sbi", "kotak", "yes",
    "rbi", "sebi", "npci", "uidai",
}

# Entity types to detect
SUPPORTED_ENTITY_TYPES = {
    "PERSON", "ORG", "LOC", "GPE",  # NER standard
    "EMAIL", "PHONE", "AADHAAR", "PAN", "UPI", "PASSPORT",  # Regex
}

# =============================================================================
# Regex Patterns (Indian PII)
# =============================================================================

_UPI_HANDLES = (
    "upi|paytm|okaxis|okicici|okhdfcbank|ybl|ibl|axl|sbi|apl|rapl|icici|"
    "hdfcbank|axisbank|kotak|indus|federal|rbl|idbi|pnb|boi|cbi|union|bob|"
    "canara|syndicate|allahabad|andhra|vijaya|dena|uco|obc|iob|mahabank|"
    "centralbank|fam|jupitermoney|slice|fi|niyobank|googlepe|phonepe|amazonpay"
)

REGEX_PATTERNS = {
    "AADHAAR": re.compile(r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]\b", re.IGNORECASE),
    "UPI": re.compile(rf"\b[a-zA-Z0-9._-]+@(?:{_UPI_HANDLES})\b", re.IGNORECASE),
    "PHONE": re.compile(r"(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}\b"),
    "EMAIL": re.compile(
        rf"\b[a-zA-Z0-9._%+-]+@(?!{_UPI_HANDLES})[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}}\b",
        re.IGNORECASE
    ),
    "PASSPORT": re.compile(r"\b[A-Z][0-9]{7}\b", re.IGNORECASE),
    "CREDIT_CARD": re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
}

# =============================================================================
# Verhoeff Checksum for Aadhaar
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


def _validate_aadhaar(aadhaar: str) -> bool:
    """Validate Aadhaar using Verhoeff algorithm."""
    digits = [int(d) for d in aadhaar if d.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


def _validate_pan(pan: str) -> bool:
    """Validate PAN format strictly."""
    pan = pan.upper()
    if len(pan) != 10:
        return False
    # Format: AAAPL1234C (4th char defines type)
    if not re.match(r"^[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]$", pan):
        return False
    return True


# =============================================================================
# NER Model (Lazy Loaded)
# =============================================================================

_ner_pipeline = None


def _ensure_ner_loaded():
    """Lazy-load the NER pipeline."""
    global _ner_pipeline
    
    if _ner_pipeline is not None:
        return
    
    try:
        from transformers import pipeline
        import torch
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("Optimum/Transformers not available, NER detection disabled")
        return
    
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    logger.info(f"Loading Lightweight NER model: {MODEL_NAME}")
    
    try:
        # Try loading Quantized ONNX model first (fast + small)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
        model = ORTModelForTokenClassification.from_pretrained(
            MODEL_NAME, 
            cache_dir=MODEL_CACHE_DIR, 
            file_name="model_quantized.onnx"
        )
        
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        logger.info("Loaded Quantized ONNX NER model (~80MB)")
        
    except Exception as e:
        logger.warning(f"Could not load ONNX model ({e}), falling back to standard...")
        # Fallback to standard pipeline if ONNX fails
        device = 0 if torch.cuda.is_available() else -1
        _ner_pipeline = pipeline(
            "ner",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            aggregation_strategy="simple",
            device=device,
            model_kwargs={"cache_dir": MODEL_CACHE_DIR},
        )
        logger.info("Loaded Standard NER model")


# =============================================================================
# Detection Functions
# =============================================================================

@dataclass
class DetectedEntity:
    """Represents a detected sensitive entity."""
    entity_type: str
    value: str
    score: float
    start: int
    end: int
    source: str  # "regex" or "ner"


def _detect_regex(text: str) -> List[DetectedEntity]:
    """Detect entities using regex patterns."""
    entities = []
    
    for entity_type, pattern in REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            
            # Validation for structured IDs
            if entity_type == "AADHAAR":
                if not _validate_aadhaar(value):
                    continue
            elif entity_type == "PAN":
                if not _validate_pan(value):
                    continue
            
            entities.append(DetectedEntity(
                entity_type=entity_type,
                value=value,
                score=1.0,  # Regex matches are high confidence
                start=match.start(),
                end=match.end(),
                source="regex",
            ))
    
    return entities


def _detect_ner(text: str, threshold: float) -> List[DetectedEntity]:
    """Detect entities using NER model."""
    _ensure_ner_loaded()
    
    if _ner_pipeline is None:
        return []
    
    entities = []
    
    try:
        results = _ner_pipeline(text)
        
        for entity in results:
            score = entity.get("score", 0)
            if score < threshold:
                continue
            
            entity_type = entity.get("entity_group", "UNKNOWN")
            value = entity.get("word", "")
            
            # Map NER labels to our types
            type_map = {
                "PER": "PERSON",
                "ORG": "ORG",
                "LOC": "LOC",
                "MISC": "MISC",
            }
            entity_type = type_map.get(entity_type, entity_type)
            
            entities.append(DetectedEntity(
                entity_type=entity_type,
                value=value,
                score=score,
                start=entity.get("start", 0),
                end=entity.get("end", 0),
                source="ner",
            ))
    except Exception as e:
        logger.error(f"NER detection error: {e}")
    
    return entities


def _filter_false_positives(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """Filter out likely false positives."""
    filtered = []
    
    for entity in entities:
        value_lower = entity.value.lower().strip()
        
        # Skip blocklisted terms
        if value_lower in _FP_BLOCKLIST:
            logger.debug(f"Filtered FP (blocklist): {entity.value}")
            continue
        
        # Skip very short entities (likely noise)
        if len(value_lower) < 3:
            continue
        
        # Skip single words that are common
        if entity.source == "ner" and entity.entity_type == "PERSON":
            # Require at least 2 tokens for person names
            if len(value_lower.split()) < 2 and len(value_lower) < 6:
                logger.debug(f"Filtered FP (short name): {entity.value}")
                continue
        
        filtered.append(entity)
    
    return filtered


def _deduplicate(entities: List[DetectedEntity]) -> List[DetectedEntity]:
    """Remove duplicate detections (prefer regex over NER)."""
    seen = set()
    unique = []
    
    # Sort by source (regex first) then by score
    sorted_entities = sorted(entities, key=lambda e: (e.source != "regex", -e.score))
    
    for entity in sorted_entities:
        key = (entity.value.lower(), entity.entity_type)
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    
    return unique


def _mask_value(value: str) -> str:
    """Mask a sensitive value for logging."""
    clean = re.sub(r"[\s\-]", "", value)
    if len(clean) > 4:
        return clean[:2] + "*" * (len(clean) - 4) + clean[-2:]
    return "****"


# =============================================================================
# Main API Function
# =============================================================================

def detect_deanonymization(
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
    use_ner: bool = True,
    redact: bool = False,
) -> List[Dict[str, Any]]:
    """
    Detect sensitive/PII data in LLM output.
    
    Args:
        text: The LLM output to scan.
        threshold: Minimum confidence for NER entities (default 0.7).
        use_ner: Whether to use NER model (default True).
        redact: Whether to redact detected entities (not implemented yet).
    
    Returns:
        List of detected issues.
    """
    if not text or not text.strip():
        return []
    
    all_entities: List[DetectedEntity] = []
    
    # Step 1: Regex detection (fast, precise)
    regex_entities = _detect_regex(text)
    all_entities.extend(regex_entities)
    
    # Step 2: NER detection (ML-based, catches names)
    if use_ner:
        ner_entities = _detect_ner(text, threshold)
        all_entities.extend(ner_entities)
    
    # Step 3: Filter false positives
    filtered = _filter_false_positives(all_entities)
    
    # Step 4: Deduplicate
    unique = _deduplicate(filtered)
    
    # Step 5: Format output
    issues = []
    for entity in unique:
        masked = _mask_value(entity.value)
        issues.append({
            "guard": "deanonymization_scanner",
            "pii_type": entity.entity_type,
            "description": f"Output contains {entity.entity_type} ({entity.source})",
            "matched": masked,
            "score": entity.score,
        })
    
    if issues:
        logger.warning(f"Detected {len(issues)} sensitive entities in output")
    
    return issues


# Pre-load model on import to ensure it's cached and ready
# This prevents downloading/loading during the first request
_ensure_ner_loaded()
