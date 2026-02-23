import os

# LOGGING
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# SECURITY
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")
if not GATEWAY_API_KEY:
    raise RuntimeError("GATEWAY_API_KEY is not set. Set it in Railway env vars.")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", GATEWAY_API_KEY)
ORIGIN_SECRET = os.getenv("ORIGIN_SECRET")
REQUIRE_CF_ACCESS_HEADERS = os.getenv("REQUIRE_CF_ACCESS_HEADERS", "0") == "1"

# APP CONFIG
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
DATABASE_URL = os.getenv("DATABASE_URL")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-0")
OPUS_MODEL = os.getenv("OPUS_MODEL", "claude-opus-4-5")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1200"))
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "250000"))
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))

MODEL_PREFIX = os.getenv("MODEL_PREFIX", "MYMODEL:")

# Token reduction knobs
STRIP_IDE_BOILERPLATE = os.getenv("STRIP_IDE_BOILERPLATE", "1") == "1"
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "20000"))
USER_MSG_MAX_CHARS = int(os.getenv("USER_MSG_MAX_CHARS", "60000"))
SYSTEM_MAX_CHARS = int(os.getenv("SYSTEM_MAX_CHARS", "40000"))
ENFORCE_DIFF_FIRST = os.getenv("ENFORCE_DIFF_FIRST", "1") == "1"

# Anthropic native prompt caching (uses cache_control blocks)
ENABLE_ANTHROPIC_CACHE_CONTROL = os.getenv("ENABLE_ANTHROPIC_CACHE_CONTROL", "1") == "1"

# Embedding / Memory
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
ENABLE_MEMORY_LAYER = os.getenv("ENABLE_MEMORY_LAYER", "0") == "1"

# Prometheus / Observability
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "1") == "1"

# Rate Limiting
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_RPM", "60"))

# Circuit Breaker
CIRCUIT_BREAKER_ENABLED = os.getenv("CIRCUIT_BREAKER_ENABLED", "1") == "1"
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))

# Retry
RETRY_ENABLED = os.getenv("RETRY_ENABLED", "1") == "1"
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "4"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "1.0"))

# Concurrency Queue
ANTHROPIC_MAX_CONCURRENCY = int(os.getenv("ANTHROPIC_MAX_CONCURRENCY", "1"))
ANTHROPIC_QUEUE_TIMEOUT = int(os.getenv("ANTHROPIC_QUEUE_TIMEOUT", "60"))

# Context Pruning
ENABLE_CONTEXT_PRUNING = os.getenv("ENABLE_CONTEXT_PRUNING", "1") == "1"
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "60000"))

# File Hash Caching
ENABLE_FILE_HASH_CACHE = os.getenv("ENABLE_FILE_HASH_CACHE", "1") == "1"
FILE_HASH_CACHE_TTL = int(os.getenv("FILE_HASH_CACHE_TTL", "3600"))

# Repo Map
ENABLE_REPO_MAP = os.getenv("ENABLE_REPO_MAP", "0") == "1"

# Smart Routing
ENABLE_SMART_ROUTING = os.getenv("ENABLE_SMART_ROUTING", "1") == "1"
OPUS_ROUTING_THRESHOLD = float(os.getenv("OPUS_ROUTING_THRESHOLD", "0.5"))

# Smart Routing v2 (local-first with LLM classifier)
SMART_ROUTING_MODE = os.getenv("SMART_ROUTING_MODE", "local_first")  # "keyword" (legacy) | "local_first" (new)
ROUTING_CLASSIFIER_MODEL = os.getenv("ROUTING_CLASSIFIER_MODEL")  # defaults to LOCAL_LLM_DEFAULT_MODEL if not set
ROUTING_CLASSIFIER_TIMEOUT = float(os.getenv("ROUTING_CLASSIFIER_TIMEOUT", "5"))  # Phase 2 timeout in seconds
ROUTING_CLASSIFIER_CACHE_SIZE = int(os.getenv("ROUTING_CLASSIFIER_CACHE_SIZE", "256"))  # LRU cache size
LOCAL_CONTEXT_CHAR_LIMIT = int(os.getenv("LOCAL_CONTEXT_CHAR_LIMIT", "30000"))  # max chars routable to local

# Cascade Routing (try local first, escalate if quality check fails)
ENABLE_CASCADE_ROUTING = os.getenv("ENABLE_CASCADE_ROUTING", "1") == "1"
CASCADE_QUALITY_CHECK_MODE = os.getenv("CASCADE_QUALITY_CHECK_MODE", "heuristic")  # "heuristic" | "llm" | "none"
CASCADE_MIN_RESPONSE_LENGTH = int(os.getenv("CASCADE_MIN_RESPONSE_LENGTH", "100"))
CASCADE_LOG_OUTCOMES = os.getenv("CASCADE_LOG_OUTCOMES", "1") == "1"
CASCADE_ELIGIBLE_TIERS = os.getenv("CASCADE_ELIGIBLE_TIERS", "local,sonnet,opus").split(",")  # Which tiers can try local first

# Opus Guard (prevent auto-escalation to most expensive model)
ALLOW_AUTO_OPUS = os.getenv("ALLOW_AUTO_OPUS", "0") == "1"  # Allow heuristic routing to auto-select Opus

# Semantic Routing Signal (embedding-based routing)
ENABLE_SEMANTIC_ROUTING_SIGNAL = os.getenv("ENABLE_SEMANTIC_ROUTING_SIGNAL", "0") == "1"
SEMANTIC_EMBEDDING_CACHE_TTL = int(os.getenv("SEMANTIC_EMBEDDING_CACHE_TTL", "3600"))

# Skills System (curated prompt templates)
ENABLE_SKILLS = os.getenv("ENABLE_SKILLS", "1") == "1"

# Batch Processing
ENABLE_BATCH_API = os.getenv("ENABLE_BATCH_API", "1") == "1"

# Plugin Tools
ENABLE_PLUGIN_TOOLS = os.getenv("ENABLE_PLUGIN_TOOLS", "1") == "1"

# Multi-project
ENABLE_MULTI_PROJECT = os.getenv("ENABLE_MULTI_PROJECT", "0") == "1"

# =============================================================================
# Local LLM Provider (Ollama via Cloudflare Access)
# =============================================================================
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL")  # e.g., https://ollama.example.com
LOCAL_LLM_DEFAULT_MODEL = os.getenv("LOCAL_LLM_DEFAULT_MODEL", "qwen2.5-coder:14b-instruct")
LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120"))

# Cloudflare Access service token headers for local LLM
LOCAL_CF_ACCESS_CLIENT_ID = os.getenv("LOCAL_CF_ACCESS_CLIENT_ID")
LOCAL_CF_ACCESS_CLIENT_SECRET = os.getenv("LOCAL_CF_ACCESS_CLIENT_SECRET")

# Allowlist of local models that can be used (security)
LOCAL_LLM_MODEL_ALLOWLIST = [
    "qwen2.5-coder:14b-instruct",
    "qwen2.5-coder:7b-instruct",
    "qwen2.5-coder:32b-instruct",
    "qwen2.5:14b",
    "qwen2.5:7b",
    "llama3.2:latest",
    "llama3.1:8b",
    "codellama:13b",
    "deepseek-coder:6.7b",
    "deepseek-coder-v2:16b",
]

# =============================================================================
# Multi-Provider API Keys (NEW)
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ollama URL (alternative to LOCAL_LLM_BASE_URL)
OLLAMA_URL = os.getenv("OLLAMA_URL", LOCAL_LLM_BASE_URL)

# =============================================================================
# Preference-Based Routing (NEW - Replaces old routing config)
# =============================================================================
# Default routing preferences (0.0 to 1.0)
# 0.0 = optimize for cost/speed, 1.0 = optimize for quality
DEFAULT_COST_QUALITY_BIAS = float(os.getenv("DEFAULT_COST_QUALITY_BIAS", "0.5"))
DEFAULT_SPEED_QUALITY_BIAS = float(os.getenv("DEFAULT_SPEED_QUALITY_BIAS", "0.5"))
DEFAULT_CASCADE_ENABLED = os.getenv("DEFAULT_CASCADE_ENABLED", "1") == "1"
DEFAULT_MAX_CASCADE_ATTEMPTS = int(os.getenv("DEFAULT_MAX_CASCADE_ATTEMPTS", "2"))

# =============================================================================
# Response Caching (NEW)
# =============================================================================
ENABLE_RESPONSE_CACHE = os.getenv("ENABLE_RESPONSE_CACHE", "1") == "1"
RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", "1800"))  # 30 minutes

# =============================================================================
# Context Compression (NEW)
# =============================================================================
ENABLE_CONTEXT_COMPRESSION = os.getenv("ENABLE_CONTEXT_COMPRESSION", "1") == "1"
COMPRESSION_MODEL = os.getenv("COMPRESSION_MODEL", "ollama/llama3.1:8b")

# =============================================================================
# Authentication (NEW)
# =============================================================================
# Enable dashboard authentication
ENABLE_DASHBOARD_AUTH = os.getenv("ENABLE_DASHBOARD_AUTH", "0") == "1"
# Secret key for session tokens (auto-generated if not set)
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", GATEWAY_API_KEY)
SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", "24"))
# Allow public registration
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "0") == "1"
