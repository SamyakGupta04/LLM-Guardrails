

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

logger = logging.getLogger("guardrails")

# Configuration

DEFAULT_OVERLAP_THRESHOLD = 0.55  # Slightly lower to allow for paraphrasing
DEFAULT_MIN_FAITHFULNESS = 0.85   # Allow some unsupported claims

# Text Processing Utilities

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP_RE = re.compile(r"^[\W_]+|[\W_]+$")

# Number patterns
_NUM_DATE_RE = re.compile(
    r"""
    (?:
        \b\d{4}-\d{2}-\d{2}\b                # 2026-02-06
      | \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b    # 06/02/2026, 6-2-26
      | \b\d+(?:\.\d+)?\s?%                  # 12%, 12.5%
      | (?:Rs\.?|₹|\$|€|£)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?  # Currency
      | \b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b     # 1,234,567.89
      | \b\d+(?:\.\d+)?\b                    # 123, 12.5
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Entity pattern (capitalized words or acronyms)
_ENTITYISH_RE = re.compile(
    r"""
    (?:
        \b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b  # New York, United Nations
      | \b[A-Z]{2,}\b                                # NASA, WHO
    )
    """,
    re.VERBOSE,
)

# Negation patterns
_NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nothing",
    "nowhere", "cannot", "can't", "couldn't", "wouldn't", "shouldn't",
    "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "hasn't", "haven't", "hadn't", "without", "lack", "lacking",
    "absent", "missing", "exclude", "excluding", "except"
}

# Contradiction indicators
_CONTRADICTION_PATTERNS = [
    (r"\bactually\b.*\bnot\b", 0.6),
    (r"\bcontrary\s+to\b", 0.7),
    (r"\bhowever\b.*\bnot\b", 0.5),
    (r"\bbut\s+not\b", 0.6),
    (r"\binstead\s+of\b", 0.5),
    (r"\brather\s+than\b", 0.4),
    (r"\bunlike\b", 0.4),
    (r"\bfalse\b", 0.7),
    (r"\bincorrect\b", 0.7),
    (r"\bwrong\b", 0.6),
]

# Number word mappings
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "lakh": 100000, "crore": 10000000, "million": 1000000, "billion": 1000000000,
}

# Entity aliases (common abbreviations)
_ENTITY_ALIASES = {
    "usa": {"united states", "us", "u.s.", "u.s.a.", "america"},
    "uk": {"united kingdom", "britain", "great britain", "england"},
    "uae": {"united arab emirates"},
    "eu": {"european union"},
    "rbi": {"reserve bank of india", "reserve bank"},
    "sebi": {"securities and exchange board of india"},
    "gst": {"goods and services tax"},
    "emi": {"equated monthly installment", "equated monthly instalment"},
    "kyc": {"know your customer"},
    "pan": {"permanent account number"},
    "ifsc": {"indian financial system code"},
    "neft": {"national electronic funds transfer"},
    "rtgs": {"real time gross settlement"},
    "imps": {"immediate payment service"},
    "upi": {"unified payments interface"},
    # Add more as needed
}

# Build reverse lookup
_ALIAS_TO_CANONICAL = {}
for canonical, aliases in _ENTITY_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical

# Stopwords for entity filtering
_ENTITY_STOPWORDS = {
    "the", "this", "that", "it", "we", "i", "in", "on", "at", "for", "from",
    "and", "or", "but", "if", "then", "because", "so", "as", "with", "by",
    "to", "of", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "shall", "also", "just", "only", "even",
    "more", "most", "other", "some", "such", "no", "not", "very", "too",
    "here", "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "many", "several", "any", "what", "which", "who", "whom"
}

# =============================================================================
# Normalization Functions
# =============================================================================

def _norm_text(s: str) -> str:
    """Normalize text for comparison."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    """Extract word tokens from text."""
    return _WORD_RE.findall(_norm_text(s))


def _token_set(s: str) -> Set[str]:
    """Get unique tokens as a set."""
    return set(_tokens(s))


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle bullet points and numbered lists
    text = re.sub(r"^\s*[-•*]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
    
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    return parts


def _strip_punct(s: str) -> str:
    """Remove leading/trailing punctuation."""
    return _PUNCT_STRIP_RE.sub("", s.strip())


def _normalize_number(s: str) -> Optional[float]:
    """
    Normalize a number string to a float.
    Handles: 1,234.56, 12.5%, Rs. 1,000, "one thousand", etc.
    """
    s = s.strip().lower()
    
    # Remove currency symbols
    s = re.sub(r"^(rs\.?|₹|\$|€|£)\s*", "", s)
    
    # Handle percentage
    is_percent = s.endswith("%")
    s = s.rstrip("%").strip()
    
    # Remove commas
    s = s.replace(",", "")
    
    # Try direct float conversion
    try:
        val = float(s)
        if is_percent:
            val = val / 100.0
        return val
    except ValueError:
        pass
    
    # Try word-based numbers
    words = s.split()
    if len(words) == 1 and words[0] in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[words[0]])
    
    # Handle compound words like "twenty five"
    if len(words) == 2:
        if words[0] in _NUMBER_WORDS and words[1] in _NUMBER_WORDS:
            return float(_NUMBER_WORDS[words[0]] + _NUMBER_WORDS[words[1]])
    
    # Handle multipliers like "5 thousand" or "2 crore"
    if len(words) == 2:
        try:
            base = float(words[0].replace(",", ""))
            if words[1] in _NUMBER_WORDS:
                return base * _NUMBER_WORDS[words[1]]
        except ValueError:
            pass
    
    return None


def _normalize_date(s: str) -> Optional[str]:
    """
    Normalize date to YYYY-MM-DD format.
    Handles various formats.
    """
    s = s.strip()
    
    # Common date formats
    formats = [
        "%Y-%m-%d",      # 2026-02-06
        "%d/%m/%Y",      # 06/02/2026
        "%m/%d/%Y",      # 02/06/2026
        "%d-%m-%Y",      # 06-02-2026
        "%d %b %Y",      # 06 Feb 2026
        "%d %B %Y",      # 06 February 2026
        "%b %d, %Y",     # Feb 06, 2026
        "%B %d, %Y",     # February 06, 2026
        "%d/%m/%y",      # 06/02/26
        "%m/%d/%y",      # 02/06/26
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return None


def _normalize_entity(s: str) -> str:
    """Normalize entity to canonical form if alias exists."""
    key = s.lower().strip()
    return _ALIAS_TO_CANONICAL.get(key, key)


def _extract_numbers(text: str) -> List[float]:
    """Extract and normalize all numbers from text."""
    matches = _NUM_DATE_RE.findall(text)
    numbers = []
    for m in matches:
        norm = _normalize_number(m)
        if norm is not None:
            numbers.append(norm)
    return numbers


def _extract_entities(text: str) -> List[str]:
    """Extract and normalize entities from text."""
    matches = _ENTITYISH_RE.findall(text)
    entities = []
    for m in matches:
        clean = _strip_punct(m)
        if clean and clean.lower() not in _ENTITY_STOPWORDS:
            entities.append(_normalize_entity(clean))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            unique.append(e)
            seen.add(e)
    return unique


# =============================================================================
# Scoring Functions
# =============================================================================

def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _containment(claim_tokens: Set[str], evidence_tokens: Set[str]) -> float:
    """Fraction of claim tokens found in evidence."""
    if not claim_tokens:
        return 1.0
    return len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))


def _lexical_score(claim: str, evidence: str) -> float:
    """
    Combined lexical similarity score.
    Weighted blend of containment and jaccard.
    """
    ct = _token_set(claim)
    et = _token_set(evidence)
    c1 = _containment(ct, et)
    c2 = _jaccard(ct, et)
    # Containment is more important for groundedness
    return 0.70 * c1 + 0.30 * c2


def _has_negation(text: str) -> bool:
    """Check if text contains negation."""
    tokens = set(_tokens(text))
    return bool(tokens & _NEGATION_WORDS)


def _check_negation_mismatch(claim: str, evidence: str) -> bool:
    """
    Check if claim and evidence have mismatched negations.
    Returns True if there's a mismatch (one has negation, other doesn't).
    """
    claim_neg = _has_negation(claim)
    evidence_neg = _has_negation(evidence)
    return claim_neg != evidence_neg


def _check_contradiction(claim: str, evidence: str) -> Tuple[bool, float]:
    """
    Check for explicit contradiction indicators.
    Returns (is_contradiction, confidence).
    """
    combined = f"{claim} {evidence}".lower()
    
    for pattern, confidence in _CONTRADICTION_PATTERNS:
        if re.search(pattern, combined):
            return True, confidence
    
    return False, 0.0


def _numbers_match(claim: str, evidence: str, tolerance: float = 0.01) -> Tuple[bool, Optional[str]]:
    """
    Check if all numbers in claim are present in evidence.
    Uses tolerance for floating point comparison.
    Returns (all_match, first_missing_number).
    """
    claim_nums = _extract_numbers(claim)
    if not claim_nums:
        return True, None
    
    evidence_nums = _extract_numbers(evidence)
    
    for cn in claim_nums:
        found = False
        for en in evidence_nums:
            if abs(cn - en) <= tolerance * max(abs(cn), abs(en), 1):
                found = True
                break
        if not found:
            return False, str(cn)
    
    return True, None


def _entities_match(claim: str, evidence: str) -> Tuple[bool, Optional[str]]:
    """
    Check if all entities in claim are present in evidence.
    Uses alias normalization.
    Returns (all_match, first_missing_entity).
    """
    claim_entities = _extract_entities(claim)
    if not claim_entities:
        return True, None
    
    evidence_text = _norm_text(evidence)
    evidence_entities = set(_extract_entities(evidence))
    
    for ce in claim_entities:
        # Check direct presence
        if ce in evidence_entities:
            continue
        # Check if normalized form is in evidence text
        if ce in evidence_text:
            continue
        # Check aliases
        found = False
        if ce in _ENTITY_ALIASES:
            for alias in _ENTITY_ALIASES[ce]:
                if alias in evidence_text:
                    found = True
                    break
        if not found:
            return False, ce
    
    return True, None


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EvidenceRef:
    """Reference to a piece of evidence."""
    chunk_id: int
    sent_id: int
    text: str
    score: float = 0.0


@dataclass
class ClaimCheck:
    """Result of checking a single claim."""
    claim: str
    supported: bool
    confidence: float
    best_evidence: Optional[EvidenceRef]
    reason: str


@dataclass
class ConsistencyResult:
    """Overall consistency check result."""
    is_consistent: bool
    faithfulness_score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: List[ClaimCheck]
    all_checks: List[ClaimCheck] = field(default_factory=list)


# =============================================================================
# Main Scanner Class
# =============================================================================

class FactualConsistencyScanner:
    """
    Factual consistency scanner for RAG output validation.
    
    Verifies that LLM output claims are grounded in the provided context.
    Uses enhanced matching with:
    - Number normalization
    - Entity alias support
    - Negation detection
    - Contradiction detection
    """
    
    def __init__(
        self,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
        require_numbers_match: bool = True,
        require_entities_match: bool = True,
        check_negations: bool = True,
        check_contradictions: bool = True,
        strict_all_claims: bool = False,
        min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS,
        ignore_conversational_fillers: bool = True,
    ):
        self.overlap_threshold = overlap_threshold
        self.require_numbers_match = require_numbers_match
        self.require_entities_match = require_entities_match
        self.check_negations = check_negations
        self.check_contradictions = check_contradictions
        self.strict_all_claims = strict_all_claims
        self.min_faithfulness = min_faithfulness
        self.ignore_conversational_fillers = ignore_conversational_fillers
        
        # Patterns for conversational fluff to ignore
        self._filler_patterns = [
            re.compile(r"^(here|this) is (a|the) (summary|report|breakdown|analysis)", re.IGNORECASE),
            re.compile(r"^based on the (context|documents|information|data)( provided)?", re.IGNORECASE),
            re.compile(r"^i (have )?provided the (details|information) below", re.IGNORECASE),
            re.compile(r"^(in|to) conclusion,?", re.IGNORECASE),
            re.compile(r"^let me (know|explain|elaborate)", re.IGNORECASE),
            re.compile(r"^please find the", re.IGNORECASE),
            re.compile(r"^hope this helps", re.IGNORECASE),
            re.compile(r"^following are the", re.IGNORECASE),
        ]
    
    def scan(self, output: str, context: str) -> ConsistencyResult:
        """
        Scan output for factual consistency with context.
        
        Args:
            output: The LLM-generated output to verify.
            context: The retrieved context/documents to check against.
        
        Returns:
            ConsistencyResult with detailed analysis.
        """
        if not output or not output.strip():
            return ConsistencyResult(
                is_consistent=True,
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
            )
        
        if not context or not context.strip():
            logger.warning("No context provided for factual consistency check")
            return ConsistencyResult(
                is_consistent=False,
                faithfulness_score=0.0,
                total_claims=1,
                supported_claims=0,
                unsupported_claims=[ClaimCheck(
                    claim=output[:100],
                    supported=False,
                    confidence=0.0,
                    best_evidence=None,
                    reason="no_context_provided"
                )],
            )
        
        # Parse context into chunks and sentences
        evidence_index = self._build_evidence_index(context)
        
        # Split output into claims
        claims = self._split_claims(output)
        
        if not claims:
            return ConsistencyResult(
                is_consistent=True,
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
            )
        
        # Verify each claim
        checks = []
        for claim in claims:
            check = self._verify_claim(claim, evidence_index)
            checks.append(check)
        
        # Calculate overall score
        supported = sum(1 for c in checks if c.supported)
        total = len(checks)
        faithfulness = supported / total if total else 1.0
        
        # Decision
        if self.strict_all_claims:
            is_consistent = (supported == total)
        else:
            is_consistent = faithfulness >= self.min_faithfulness
        
        unsupported = [c for c in checks if not c.supported]
        
        if not is_consistent:
            logger.warning(
                f"Factual consistency check failed: "
                f"{supported}/{total} claims supported, "
                f"faithfulness={faithfulness:.2%}"
            )
        
        return ConsistencyResult(
            is_consistent=is_consistent,
            faithfulness_score=faithfulness,
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            all_checks=checks,
        )
    
    def _build_evidence_index(self, context: str) -> List[EvidenceRef]:
        """Build sentence-level evidence index from context."""
        index = []
        
        # Try to parse structured format [C0], [C1], etc.
        chunks = self._parse_structured_context(context)
        
        if not chunks:
            # Fallback: treat entire context as one chunk
            chunks = [(0, context)]
        
        for chunk_id, chunk_text in chunks:
            sentences = _split_sentences(chunk_text)
            for sent_id, sent in enumerate(sentences):
                sent_clean = sent.strip()
                if sent_clean and len(sent_clean) > 5:
                    index.append(EvidenceRef(
                        chunk_id=chunk_id,
                        sent_id=sent_id,
                        text=sent_clean,
                    ))
        
        return index
    
    def _parse_structured_context(self, context: str) -> List[Tuple[int, str]]:
        """
        Parse structured context with [C0], [C1], etc. markers.
        Also handles CONTEXT: prefix.
        """
        # Try to find CONTEXT: marker
        marker_match = re.search(r"CONTEXT\s*:", context, re.IGNORECASE)
        if marker_match:
            context = context[marker_match.end():].strip()
        
        chunks = []
        current_id: Optional[int] = None
        buf: List[str] = []
        
        for line in context.splitlines():
            line_strip = line.rstrip()
            
            # Check for chunk header [C#] or [Chunk #] or similar
            m = re.match(r"^\s*\[C(?:hunk)?(\d+)\]\s*(.*)$", line_strip, re.IGNORECASE)
            if m:
                # Flush previous chunk
                if current_id is not None and buf:
                    chunks.append((current_id, "\n".join(buf).strip()))
                current_id = int(m.group(1))
                buf = [m.group(2)] if m.group(2) else []
            else:
                if current_id is not None:
                    buf.append(line)
        
        # Flush last chunk
        if current_id is not None and buf:
            chunks.append((current_id, "\n".join(buf).strip()))
        
        return [(cid, txt) for cid, txt in chunks if txt]
    
    def _split_claims(self, output: str) -> List[str]:
        """
        Split output into atomic claims for verification.
        Handles sentences, semicolons, and compound statements.
        """
        # First split by sentences
        sentences = _split_sentences(output)
        
        claims = []
        for sent in sentences:
            # Split on semicolons
            parts = [p.strip() for p in sent.split(";") if p.strip()]
            
            for part in parts:
                # Optionally split on " and " for compound claims
                if " and " in part.lower():
                    sub_claims = self._split_compound_claim(part)
                    claims.extend(sub_claims)
                else:
                    claims.append(part)
        
        # Filter out very short claims (likely not factual)
        claims = [c.strip() for c in claims if len(c.strip()) >= 10]
        
        return claims
    
    def _split_compound_claim(self, text: str) -> List[str]:
        """Split compound claims joined by 'and'."""
        parts = re.split(r"\s+\band\b\s+", text, flags=re.IGNORECASE)
        
        if len(parts) <= 1:
            return [text]
        
        # Only split if both parts are substantial
        if any(len(p.strip()) < 8 for p in parts):
            return [text]
        
        return [p.strip() for p in parts if p.strip()]
    
    def _verify_claim(self, claim: str, evidence_index: List[EvidenceRef]) -> ClaimCheck:
        """Verify a single claim against evidence."""
        
        # Check for conversational fluff
        if self.ignore_conversational_fillers:
            for pattern in self._filler_patterns:
                if pattern.search(claim):
                    return ClaimCheck(
                        claim=claim,
                        supported=True,
                        confidence=1.0,
                        best_evidence=None,
                        reason="conversational_filler_ignored",
                    )

        # Find best matching evidence
        best_evidence = None
        best_score = 0.0
        
        # Pre-calculate claim features
        claim_entities = _extract_entities(claim)
        claim_numbers = _extract_numbers(claim)
        
        for e in evidence_index:
            score = _lexical_score(claim, e.text)
            
            # --- Entity/Number Boost ---
            # If standard overlap is low but unique facts match perfectly, boost score.
            # This helps with paraphrasing (e.g. "car" vs "vehicle") if the ID/Price matches.
            valid_entities = [ent for ent in claim_entities if ent in _norm_text(e.text)]
            valid_numbers = [num for num in claim_numbers if _numbers_match(str(num), e.text)[0]]
            
            if len(valid_entities) >= 1 or len(valid_numbers) >= 1:
                # Boost based on facts matched
                boost = 0.0
                if len(valid_entities) > 0: boost += 0.15
                if len(valid_numbers) > 0: boost += 0.15
                
                # Apply boost but cap at 0.95 (still need some lexical match)
                if score > 0.1: # Minimum relevance floor
                    score = min(0.95, score + boost)

            if score > best_score:
                best_score = score
                best_evidence = EvidenceRef(
                    chunk_id=e.chunk_id,
                    sent_id=e.sent_id,
                    text=e.text,
                    score=score,
                )
        
        if not best_evidence:
            return ClaimCheck(
                claim=claim,
                supported=False,
                confidence=0.0,
                best_evidence=None,
                reason="no_evidence_found",
            )
        
        # Check overlap threshold
        # Allow lower threshold if we had strong entity matches (implied by high boosted score)
        effective_threshold = self.overlap_threshold
        if best_evidence.score > best_score: # It was boosted
             effective_threshold -= 0.1

        if best_evidence.score < effective_threshold:
            return ClaimCheck(
                claim=claim,
                supported=False,
                confidence=best_evidence.score,
                best_evidence=best_evidence,
                reason=f"low_overlap({best_evidence.score:.2f}<{effective_threshold:.2f})",
            )
        
        # Check negation mismatch
        if self.check_negations:
            if _check_negation_mismatch(claim, best_evidence.text):
                return ClaimCheck(
                    claim=claim,
                    supported=False,
                    confidence=best_evidence.score * 0.5,
                    best_evidence=best_evidence,
                    reason="negation_mismatch",
                )
        
        # Check for contradictions
        if self.check_contradictions:
            is_contradiction, conf = _check_contradiction(claim, best_evidence.text)
            if is_contradiction:
                return ClaimCheck(
                    claim=claim,
                    supported=False,
                    confidence=1.0 - conf,
                    best_evidence=best_evidence,
                    reason=f"contradiction_detected(conf={conf:.2f})",
                )
        
        # Check number matching
        if self.require_numbers_match:
            match, missing = _numbers_match(claim, best_evidence.text)
            if not match:
                return ClaimCheck(
                    claim=claim,
                    supported=False,
                    confidence=best_evidence.score * 0.7,
                    best_evidence=best_evidence,
                    reason=f"number_mismatch(missing:{missing})",
                )
        
        # Check entity matching
        if self.require_entities_match:
            match, missing = _entities_match(claim, best_evidence.text)
            if not match:
                return ClaimCheck(
                    claim=claim,
                    supported=False,
                    confidence=best_evidence.score * 0.8,
                    best_evidence=best_evidence,
                    reason=f"entity_mismatch(missing:{missing})",
                )
        
        # All checks passed
        return ClaimCheck(
            claim=claim,
            supported=True,
            confidence=best_evidence.score,
            best_evidence=best_evidence,
            reason="supported",
        )


# =============================================================================
# API Function
# =============================================================================

def detect_factual_inconsistency(
    output: str,
    context: str = "",
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    require_numbers_match: bool = True,
    require_entities_match: bool = True,
    check_negations: bool = True,
    check_contradictions: bool = True,
    strict_all_claims: bool = False,
    min_faithfulness: float = DEFAULT_MIN_FAITHFULNESS,
) -> List[Dict[str, Any]]:
    """
    Detect factual inconsistencies in output against context.
    
    Args:
        output: The LLM-generated text to verify.
        context: The source context/documents to check against.
        overlap_threshold: Minimum lexical overlap for support.
        require_numbers_match: Require all numbers to match.
        require_entities_match: Require all entities to match.
        check_negations: Check for negation mismatches.
        check_contradictions: Check for contradiction indicators.
        strict_all_claims: Require all claims to be supported.
        min_faithfulness: Minimum ratio of supported claims.
    
    Returns:
        List of issues found.
    """
    if not output or not output.strip():
        return []
    
    if not context or not context.strip():
        # Return warning but don't block if no context provided
        logger.warning("No context provided for factual consistency check")
        return []
    
    scanner = FactualConsistencyScanner(
        overlap_threshold=overlap_threshold,
        require_numbers_match=require_numbers_match,
        require_entities_match=require_entities_match,
        check_negations=check_negations,
        check_contradictions=check_contradictions,
        strict_all_claims=strict_all_claims,
        min_faithfulness=min_faithfulness,
    )
    
    result = scanner.scan(output, context)
    
    if result.is_consistent:
        return []
    
    issues = []
    
    # Main issue
    issues.append({
        "guard": "factual_consistency_scanner",
        "description": f"Output may not be factually consistent with context "
                      f"({result.supported_claims}/{result.total_claims} claims supported, "
                      f"{result.faithfulness_score:.0%} faithfulness)",
        "score": 1.0 - result.faithfulness_score,
        "faithfulness": result.faithfulness_score,
        "total_claims": result.total_claims,
        "supported_claims": result.supported_claims,
    })
    
    # Add details about unsupported claims (limit to top 3)
    for check in result.unsupported_claims[:3]:
        issues.append({
            "guard": "factual_consistency_scanner",
            "description": f"Unsupported claim: {check.reason}",
            "matched": check.claim[:100] + "..." if len(check.claim) > 100 else check.claim,
            "score": 1.0 - check.confidence,
            "best_evidence": check.best_evidence.text[:100] if check.best_evidence else None,
        })
    
    return issues
