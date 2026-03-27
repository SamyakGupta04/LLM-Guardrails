"""
Malicious URL Detection Guard
Heuristic-based scanner for malicious URLs in output.
"""
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Sequence

import logging

logger = logging.getLogger("guardrails")

# =============================================================================
# Configuration
# =============================================================================

THRESHOLD = 0.65

SUSPICIOUS_TLDS = {
    "zip", "mov", "top", "xyz", "click", "country", "monster", "gq", "tk", "icu"
}
SUSPICIOUS_PATH_TOKENS = ("/download", "/payload", "/exe", "/apk", "/login", "/verify", "/update")

# Regex for trailing punctuation removal
_TRAILING_PUNCT_RE = re.compile(r"[)\].,;:!?]+$")

# Regex for finding URLs (simplified but effective)
_URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# =============================================================================
# Helper Functions
# =============================================================================

def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    return _URL_PATTERN.findall(text)

def _clean_url(u: str) -> str:
    """Clean trailing punctuation from URL."""
    u = u.strip()
    u = _TRAILING_PUNCT_RE.sub("", u)
    return u

def _hostname(u: str) -> str:
    """Extract hostname from URL."""
    try:
        p = urlparse(u if "://" in u else "http://" + u)
        return (p.hostname or "").lower()
    except Exception:
        return ""

def _tld(host: str) -> str:
    """Extract Top Level Domain."""
    parts = host.split(".")
    return parts[-1] if len(parts) >= 2 else ""

def _is_ip_host(host: str) -> bool:
    """Check if host is an IP address."""
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))

def _unique_preserve_order(items: Sequence[str]) -> List[str]:
    """Deduplicate list while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _has_credential_in_url(u: str) -> bool:
    """Check for credentials in URL."""
    try:
        p = urlparse(u if "://" in u else "http://" + u)
        return bool(p.username or p.password)
    except Exception:
        return False

def _suspicion_score(url: str) -> float:
    """
    Heuristic suspicion score in [0,1].
    """
    u = url.lower()
    host = _hostname(url)
    tld = _tld(host)

    score = 0.0

    # IP-based URLs are often used for malware delivery
    if host and _is_ip_host(host):
        score = max(score, 0.7)

    # Punycode domains can be used for homograph attacks
    if "xn--" in host:
        score = max(score, 0.6)

    # Suspicious TLDs
    if tld in SUSPICIOUS_TLDS:
        score = max(score, 0.55)

    # Credentials in URL are highly suspicious
    if _has_credential_in_url(url):
        score = max(score, 0.85)

    # Suspicious path tokens
    if any(tok in u for tok in SUSPICIOUS_PATH_TOKENS):
        score = max(score, 0.65)

    # Non-standard ports
    try:
        p = urlparse(url if "://" in url else "http://" + url)
        if p.port and p.port not in (80, 443):
            score = max(score, 0.55)
    except Exception:
        pass

    return min(score, 1.0)

# =============================================================================
# Main Detection Function
# =============================================================================

def detect_malicious_urls(text: str) -> List[Dict[str, Any]]:
    """
    Detect malicious URLs in the output text using heuristics.
    """
    if not text or not text.strip():
        return []

    issues = []
    urls = extract_urls(text)
    
    if not urls:
        return []

    urls = [_clean_url(u) for u in urls]
    urls = [u for u in urls if u]
    urls = _unique_preserve_order(urls)

    for u in urls:
        score = _suspicion_score(u)
        
        # Using >= THRESHOLD to catch edge cases like path tokens (0.65)
        if score >= THRESHOLD:
            issues.append({
                "guard": "malicious_url_detector",
                "description": f"Suspicious URL detected",
                "matched": u,
                "score": score
            })
            
    return issues
