"""
Competitor Name Detection Guard (Simplified)

Detects mentions of competitor bank names in user prompts using:
1. Simple keyword matching for strong aliases
2. Gemini LLM for disambiguation of weak/ambiguous matches

Returns issues if competitor mentions are detected.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guardrails.competitor")


# =============================================================================
# Configuration
# =============================================================================

COMPETITOR_THRESHOLD = 0.70
CACHE_TTL_S = 300


# =============================================================================
# Competitor Bank List (India - excludes Federal Bank, FedFina)
# =============================================================================

COMPETITOR_BANKS = [
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Mahindra Bank",
    "IndusInd Bank",
    "Yes Bank",
    "IDFC First Bank",
    "IDBI Bank",
    "Bandhan Bank",
    "RBL Bank",
    "South Indian Bank",
    "Karnataka Bank",
    "Karur Vysya Bank",
    "City Union Bank",
    "DCB Bank",
    "AU Small Finance Bank",
    "Equitas Small Finance Bank",
    "Ujjivan Small Finance Bank",
]


# =============================================================================
# Alias Registry - Simplified
# Strong: Always match (exact or near-exact bank names)
# Weak: Require LLM confirmation (ambiguous terms)
# =============================================================================

STRONG_ALIASES: Dict[str, List[str]] = {
    "State Bank of India": ["state bank of india", "sbi bank", "sbi"],
    "HDFC Bank": ["hdfc bank", "hdfc"],
    "ICICI Bank": ["icici bank", "icici"],
    "Axis Bank": ["axis bank", "axisbank"],
    "Kotak Mahindra Bank": ["kotak mahindra bank", "kotak bank", "kotak"],
    "IndusInd Bank": ["indusind bank", "indusind"],
    "Yes Bank": ["yes bank"],
    "IDFC First Bank": ["idfc first bank", "idfc first", "idfc bank", "idfc"],
    "IDBI Bank": ["idbi bank", "idbi"],
    "Bandhan Bank": ["bandhan bank", "bandhan"],
    "RBL Bank": ["rbl bank", "rbl"],
    "South Indian Bank": ["south indian bank"],
    "Karnataka Bank": ["karnataka bank"],
    "Karur Vysya Bank": ["karur vysya bank", "kvb"],
    "City Union Bank": ["city union bank", "cub"],
    "DCB Bank": ["dcb bank", "dcb"],
    "AU Small Finance Bank": ["au small finance bank", "au bank", "au sfb"],
    "Equitas Small Finance Bank": ["equitas small finance bank", "equitas bank", "equitas"],
    "Ujjivan Small Finance Bank": ["ujjivan small finance bank", "ujjivan bank", "ujjivan"],
}

# Weak aliases need LLM disambiguation
WEAK_ALIASES: Dict[str, List[str]] = {
    "Axis Bank": ["axis"],  # Could be "y axis", "x-axis"
    "Yes Bank": ["yes"],    # Could be affirmation
    "AU Small Finance Bank": ["au"],  # Could be Australia
}


# =============================================================================
# Simple Cache
# =============================================================================

class _SimpleCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        ts, val = item
        if (time.time() - ts) > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        self._store[key] = (time.time(), val)


_cache = _SimpleCache(ttl_seconds=CACHE_TTL_S)


# =============================================================================
# Text Normalization with Deobfuscation
# =============================================================================

def collapse_spaced_text(text: str) -> str:
    """
    Collapse obfuscated text like "A x i s" or "I C I C I" to "Axis" or "ICICI".
    Matches patterns where single letters are separated by spaces.
    """
    # Pattern: single letters separated by spaces (at least 3 letters)
    # e.g., "A x i s" -> "Axis", "I C I C I" -> "ICICI"
    def collapse_match(m):
        chars = m.group(0).replace(" ", "")
        return chars
    
    # Match: letter space letter space letter... (at least 3 letters)
    pattern = r'\b[A-Za-z](?:\s+[A-Za-z]){2,}\b'
    text = re.sub(pattern, collapse_match, text)
    return text


def normalize(text: str) -> str:
    """Normalize text: collapse obfuscation, lowercase, collapse whitespace."""
    # First, collapse spaced out letters like "A x i s" -> "Axis"
    text = collapse_spaced_text(text)
    # Then lowercase and normalize whitespace
    return " ".join(text.lower().split())


def find_strong_matches(text: str) -> Set[str]:
    """Find competitors with strong alias matches."""
    text_norm = normalize(text)
    found: Set[str] = set()
    
    for competitor, aliases in STRONG_ALIASES.items():
        for alias in aliases:
            # Use word boundary matching
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_norm):
                found.add(competitor)
                print(f"DEBUG: Strong match found: '{alias}' -> {competitor}")
                break
    
    return found


def find_weak_matches(text: str) -> List[Tuple[str, str]]:
    """Find potential weak alias matches that need disambiguation."""
    text_norm = normalize(text)
    found: List[Tuple[str, str]] = []
    
    for competitor, aliases in WEAK_ALIASES.items():
        for alias in aliases:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_norm):
                found.append((competitor, alias))
                print(f"DEBUG: Weak match found: '{alias}' -> {competitor}")
    
    return found


# Banking context keywords - if these are present, weak aliases are likely bank references
BANKING_CONTEXT_KEYWORDS = {
    "account", "accounts", "bank", "banking", "loan", "loans", "emi", "emis",
    "ifsc", "branch", "branches", "netbanking", "upi", "imps", "neft", "rtgs",
    "credit", "debit", "card", "cards", "transfer", "balance", "fd", "fixed deposit",
    "rd", "recurring", "savings", "current", "atm", "cheque", "check", "statement",
    "interest", "rate", "rates", "deposit", "withdrawal", "passbook", "kyc",
    "mobile banking", "internet banking", "transaction", "transactions",
}


def has_banking_context(text: str) -> bool:
    """Check if text contains banking-related keywords."""
    text_lower = text.lower()
    words = set(text_lower.split())
    return bool(words.intersection(BANKING_CONTEXT_KEYWORDS))


# =============================================================================
# LLM Disambiguation
# =============================================================================

def disambiguate_with_llm(prompt: str, weak_matches: List[Tuple[str, str]]) -> Set[str]:
    """Use Gemini to disambiguate weak matches."""
    if not weak_matches:
        return set()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("DEBUG: No API key for LLM disambiguation, skipping weak matches")
        return set()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Build prompt for LLM
    candidates = [{"competitor": c, "alias": a} for c, a in weak_matches]
    
    llm_prompt = f"""Analyze this text and determine if the ambiguous terms refer to BANKS or not.

TEXT: "{prompt}"

CANDIDATES TO CHECK:
{json.dumps(candidates, indent=2)}

RULES:
- "axis" in math context (y axis, x-axis, axis of graph) = NOT a bank
- "axis" in banking context (Axis account, Axis loan) = BANK
- "yes" as simple affirmation = NOT a bank
- "yes" in "Yes Bank" context = BANK
- "au" as country code = NOT a bank
- "au" in banking context = BANK

Return JSON only:
{{"matches": ["competitor_name_if_match", ...]}}

If no matches, return: {{"matches": []}}"""

    try:
        response = model.generate_content(
            llm_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        
        data = json.loads(response.text.strip())
        matches = set(data.get("matches", []))
        print(f"DEBUG: LLM returned matches: {matches}")
        return matches
        
    except Exception as e:
        print(f"DEBUG: LLM error: {e}")
        return set()


# =============================================================================
# Main Detection Function
# =============================================================================

def detect_competitors(prompt: str) -> List[Dict[str, Any]]:
    """
    Detect competitor bank mentions in a prompt.
    Returns list of issues if competitors are detected.
    """
    if not prompt or not prompt.strip():
        return []

    # Check cache
    cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    print(f"DEBUG: Scanning for competitors in: '{prompt}'")

    issues: List[Dict[str, Any]] = []

    # Stage 1: Find strong matches (auto-confirm)
    confirmed = find_strong_matches(prompt)

    # Stage 2: Find weak matches and disambiguate
    weak_matches = find_weak_matches(prompt)
    
    # Filter out weak matches for competitors already confirmed
    weak_matches = [(c, a) for c, a in weak_matches if c not in confirmed]
    
    if weak_matches:
        # Check if there's banking context in the prompt
        banking_context = has_banking_context(prompt)
        print(f"DEBUG: Banking context present: {banking_context}")
        
        if banking_context:
            # If banking context is present, auto-confirm weak matches
            for competitor, alias in weak_matches:
                print(f"DEBUG: Auto-confirming weak match due to banking context: '{alias}' -> {competitor}")
                confirmed.add(competitor)
        else:
            # No banking context - use LLM for disambiguation
            llm_confirmed = disambiguate_with_llm(prompt, weak_matches)
            confirmed.update(llm_confirmed)

    print(f"DEBUG: Final confirmed competitors: {confirmed}")

    # Build issue if competitors found
    if confirmed:
        competitors_list = sorted(confirmed)
        description = f"Competitor bank mention detected: {', '.join(competitors_list)}"

        issues.append({
            "guard": "competitor_detector",
            "description": description,
            "matched": ", ".join(competitors_list),
            "competitors": competitors_list,
            "score": 1.0,
        })

    _cache.set(cache_key, issues)
    return issues


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    test_cases = [
        "sbi bank",
        "axis bank",
        "HDFC Bank",
        "I want to compare SBI and ICICI",
        "Plot data on the y axis",
        "Yes, I want to proceed",
        "Open account with Yes Bank",
        # Obfuscation test cases
        "Move to A x i s Bank and enable UPI",
        "I C I C I Bank netbanking is down",
        "H D F C has good rates",
        "S B I offers fixed deposit",
        # Banking context with weak aliases
        "Open an account in AU and share IFSC",
        "Transfer money via Axis UPI",
        "Check AU loan rates",
        # Non-banking context (should pass)
        "AU is Australia country code",
        "What is the weather today",
    ]
    
    for test in test_cases:
        print(f"\n{'='*50}")
        print(f"Input: {test}")
        result = detect_competitors(test)
        if result:
            print(f"BLOCKED: {result[0]['competitors']}")
        else:
            print("ALLOWED")
