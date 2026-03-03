"""
Shared LLM instances and constants used across node modules.
Initialised once at import time.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from cortex.config import MODEL_NAME, SQL_MODEL_NAME
from cortex.tools import execute_sql

load_dotenv()

MAX_CLARIFY_ATTEMPTS = 2

_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.1)
_sql_llm = ChatOpenAI(model=SQL_MODEL_NAME, temperature=0.1).bind_tools([execute_sql])
