"""
Shared LLM instances, constants, and error-handling utilities used across node modules.
Initialised once at import time.
"""

import random
import time

import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from cortex.config import MODEL_NAME, SQL_MODEL_NAME
from cortex.tools import execute_sql

load_dotenv()

MAX_CLARIFY_ATTEMPTS = 2

_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.1)
_sql_llm = ChatOpenAI(model=SQL_MODEL_NAME, temperature=0.1).bind_tools([execute_sql])


def _is_transient(e: Exception) -> bool:
    """True for errors worth retrying: timeouts, connection failures, 408/429/5xx."""
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code in (408, 429, 500, 502, 503, 504)
    return False


def sanitize_error(e: Exception) -> str:
    """Return type name + first line only — no keys, SQL, or PII. Max ~100 chars."""
    first_line = str(e).split("\n")[0][:100]
    return f"{type(e).__name__}: {first_line}"


def invoke_with_retry(call, *, retries: int = 1, base_delay: float = 1.0):
    """Call call() and retry only on transient API errors (rate limit, timeout, 5xx).

    Non-transient errors (schema mismatch, bad request) are re-raised immediately.
    Raises last exception after retries are exhausted.
    """
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as e:
            if not _is_transient(e):
                raise
            last_exc = e
            if attempt < retries:
                time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.3))
    raise last_exc
