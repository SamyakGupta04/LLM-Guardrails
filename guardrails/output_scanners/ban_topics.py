"""
Ban Topics Scanner
Detects and blocks banned topics using semantic similarity + keyword triangulation.

Features:
- Reuses existing embedding model (all-MiniLM-L6-v2)
- Zero new model downloads
- Hybrid detection:
  1. Base Topic Similarity (Broad check)
  2. Intent/Harm Similarity (Specific check)
  3. Keyword Density (Booster)
"""

import logging
from typing import List, Dict, Any, Optional, Set
import numpy as np

# Reuse the embedding logic from relevance scanner to avoid code duplication
# We import _encode and _compute_similarity if available, or re-implement wrappers
try:
    from output_scanners.relevance import _encode, compute_similarity, _ensure_model_loaded
except ImportError:
    # Fallback if relevance scanner not properly initialized
    logging.getLogger("guardrails").warning("Relevance scanner modules not found. BanTopics may fail.")
    pass

logger = logging.getLogger("guardrails")

# =============================================================================
# Configuration
# =============================================================================

# Default Thresholds
DEFAULT_THRESHOLD = 0.60       # Base threshold for topic similarity
HIGH_CONFIDENCE_THRESHOLD = 0.75 # Threshold for purely semantic matches

# 1. BANNED TOPICS (Broad Categories)
# These are used to check if the output is generally "about" these topics
BANNED_TOPICS_BROAD = [
    "politics and government policy",
    "religious controversy and theology",
    "sexual content and nsfw",
    "violence and terrorism",
    "hate speech and discrimination",
    "criminal activities and illegal acts",
]

# 2. HARMFUL INTENTS (Specific Negative Contexts)
# These are used to differentiate "news about politics" vs "political propaganda"
BANNED_INTENTS = [
    "criticism of government or political parties",
    "promoting religious hatred or intolerance",
    "encouraging violence or insurrection",
    "sexually explicit or pornographic description",
    "discriminatory slurs against a group",
    "instructions for illegal activities",
    "political propaganda or campaigning",
]

# 3. KEYWORD BLOCKLIST (Fast Fail / Boosters)
# Presence of these words lowers the threshold significantly
KEYWORD_BLOCKLIST = {
    # Hate / Violence
    "kill", "murder", "terrorist", "bomb", "blast", "attack", "slaughter",
    "genocide", "racist", "sexist", "nazi", "hitler", "jihad", "crusade",
    # Indian Political Sensitivity (Context dependent)
    "khalistan", "naxal", "naxalite", "maoist", "insurgent", "separatist",
    "anti-national", "traitor", "sedition",
    # NSFW
    "porn", "xxx", "nude", "naked", "sex",
}

# Pre-computed embeddings cache
_topic_embeddings = {}
_intent_embeddings = {}
_initialized = False

def _initialize_scanner():
    """Pre-compute embeddings for banned topics on first run."""
    global _initialized, _topic_embeddings, _intent_embeddings
    
    if _initialized:
        return

    try:
        _ensure_model_loaded()
        
        logger.info("Initializing BanTopics scanner (pre-computing embeddings)...")
        
        # Compute embeddings for broad topics
        for topic in BANNED_TOPICS_BROAD:
            _topic_embeddings[topic] = _encode(topic)
            
        # Compute embeddings for specific intents
        for intent in BANNED_INTENTS:
            _intent_embeddings[intent] = _encode(intent)
            
        _initialized = True
        logger.info("BanTopics scanner initialized.")
        
    except Exception as e:
        logger.error(f"Failed to initialize BanTopics scanner: {e}")

def detect_banned_topics(
    output: str,
    threshold: float = DEFAULT_THRESHOLD,
    topics: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Detect if output contains banned topics.
    
    Args:
        output: Text to scan.
        threshold: Sensitivity threshold (0.0 - 1.0).
        topics: Optional list of custom topics to add to the check.
    
    Returns:
        List of detected issues.
    """
    if not output or not output.strip():
        return []
    
    _initialize_scanner()
    if not _initialized:
        return []

    issues = []
    
    # 1. Keyword Check (Fast Fail & Threshold Adjustment)
    output_lower = output.lower()
    keyword_matches = [word for word in KEYWORD_BLOCKLIST if word in output_lower]
    
    # If keywords found, lower the threshold (make it stricter/easier to trigger)
    effective_threshold = threshold
    if keyword_matches:
        effective_threshold = max(0.4, threshold - 0.15) # Lower threshold by 0.15
        logger.debug(f"Keywords found ({keyword_matches}). Lowering threshold to {effective_threshold}")

    # 2. Semantic Analysis
    try:
        output_embedding = _encode(output)
        
        # Check against Broad Topics
        max_topic_score = 0.0
        best_topic = None
        
        for topic, emb in _topic_embeddings.items():
            score = float(np.dot(output_embedding, emb))
            if score > max_topic_score:
                max_topic_score = score
                best_topic = topic
        
        # Check against Specific Intents (Harmful context)
        max_intent_score = 0.0
        best_intent = None
        
        for intent, emb in _intent_embeddings.items():
            score = float(np.dot(output_embedding, emb))
            if score > max_intent_score:
                max_intent_score = score
                best_intent = intent
        
        # Triangulation Logic
        # We flag if:
        # A) High Semantic Match (> High Threshold)
        # B) Moderate Semantic Match + Keywords (> Effective Threshold)
        # C) Moderate Topic Match + Moderate Intent Match (Combined context)
        
        is_banned = False
        reason = ""
        
        # Case A: High Confidence Semantic Match
        if max_intent_score > HIGH_CONFIDENCE_THRESHOLD:
            is_banned = True
            reason = f"High matching intent: '{best_intent}' (Score: {max_intent_score:.2f})"
            
        # Case B: Threshold Check (with potential keyword lowering)
        elif max_intent_score > effective_threshold:
            is_banned = True
            reason = f"Matching intent: '{best_intent}' (Score: {max_intent_score:.2f} > {effective_threshold:.2f})"
        
        # Case C: Combined Context (Topic + Intent both reasonable)
        elif max_topic_score > 0.6 and max_intent_score > 0.55:
            is_banned = True
            reason = f"Combined Context: Topic '{best_topic}' ({max_topic_score:.2f}) + Intent '{best_intent}' ({max_intent_score:.2f})"

        if is_banned:
            issues.append({
                "guard": "ban_topics_scanner",
                "description": f"Output detected as Banned Content. {reason}",
                "score": max_intent_score,
                "matched": best_intent
            })
            
            logger.warning(f"Banned Topic Detected: {reason}")
            
    except Exception as e:
        logger.error(f"Error in BanTopics analysis: {e}")
        
    return issues
