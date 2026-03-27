"""
Token Limit Detection Guard

Checks if a prompt exceeds a specific token limit.
Uses tiktoken library for accurate token counting.

Returns issues if prompt exceeds the token limit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guardrails.token_limit")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TOKEN_LIMIT = 4096
DEFAULT_ENCODING = "cl100k_base"  # GPT-4, GPT-3.5-turbo encoding


# =============================================================================
# Token Counter
# =============================================================================

_encoding = None


def _get_encoding():
    """Lazy load tiktoken encoding."""
    global _encoding
    if _encoding is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
            logger.info(f"Loaded tiktoken encoding: {DEFAULT_ENCODING}")
        except ImportError:
            logger.warning("tiktoken not installed. Using approximate token counting.")
            _encoding = "approximate"
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or approximate method."""
    encoding = _get_encoding()
    
    if encoding == "approximate":
        # Approximate: ~4 chars per token for English
        return len(text) // 4
    
    return len(encoding.encode(text))


def split_text_on_tokens(text: str, limit: int) -> tuple[List[str], int]:
    """Split text into chunks that fit within token limit."""
    encoding = _get_encoding()
    
    if encoding == "approximate":
        # Approximate splitting
        char_limit = limit * 4
        chunks = []
        for i in range(0, len(text), char_limit):
            chunks.append(text[i:i + char_limit])
        return chunks, len(text) // 4
    
    input_ids = encoding.encode(text)
    total_tokens = len(input_ids)
    
    if total_tokens <= limit:
        return [text], total_tokens
    
    chunks: List[str] = []
    start_idx = 0
    
    while start_idx < len(input_ids):
        cur_idx = min(start_idx + limit, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        chunks.append(encoding.decode(chunk_ids))
        start_idx += limit
    
    return chunks, total_tokens


# =============================================================================
# Main Detection Function
# =============================================================================

def detect_token_limit(
    prompt: str,
    limit: int = DEFAULT_TOKEN_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Check if prompt exceeds token limit.
    
    Parameters:
        prompt: The input text to check
        limit: Maximum allowed tokens (default 4096)
    
    Returns:
        List of issues if limit exceeded, empty list otherwise.
    """
    if not prompt or not prompt.strip():
        return []

    issues: List[Dict[str, Any]] = []
    
    num_tokens = count_tokens(prompt)
    
    if num_tokens <= limit:
        logger.debug(f"Token count OK: {num_tokens}/{limit}")
        return []
    
    # Token limit exceeded
    chunks, _ = split_text_on_tokens(prompt, limit)
    
    logger.warning(f"Token limit exceeded: {num_tokens}/{limit} tokens")
    
    issues.append({
        "guard": "token_limit",
        "description": f"Prompt exceeds token limit ({num_tokens}/{limit} tokens)",
        "matched": f"First {min(50, len(prompt))} chars: {prompt[:50]}...",
        "score": 1.0,
        "num_tokens": num_tokens,
        "limit": limit,
        "num_chunks": len(chunks),
    })
    
    return issues


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test with short text
    short_text = "Hello, how are you?"
    print(f"\nTesting: '{short_text}'")
    result = detect_token_limit(short_text, limit=100)
    print(f"Tokens: {count_tokens(short_text)}")
    print(f"Issues: {result}")
    
    # Test with long text
    long_text = "Hello world. " * 1000  # ~3000 tokens approximately
    print(f"\nTesting long text ({len(long_text)} chars)")
    result = detect_token_limit(long_text, limit=100)
    print(f"Tokens: {count_tokens(long_text)}")
    print(f"Issues: {result}")
