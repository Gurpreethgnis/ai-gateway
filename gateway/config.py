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
RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
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

# Batch Processing
ENABLE_BATCH_API = os.getenv("ENABLE_BATCH_API", "1") == "1"

# Plugin Tools
ENABLE_PLUGIN_TOOLS = os.getenv("ENABLE_PLUGIN_TOOLS", "1") == "1"

# Multi-project
ENABLE_MULTI_PROJECT = os.getenv("ENABLE_MULTI_PROJECT", "0") == "1"
