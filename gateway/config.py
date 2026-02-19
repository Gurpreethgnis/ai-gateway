import os

# LOGGING
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# SECURITY
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY")
if not GATEWAY_API_KEY:
    raise RuntimeError("GATEWAY_API_KEY is not set. Set it in Railway env vars.")

ORIGIN_SECRET = os.getenv("ORIGIN_SECRET")
REQUIRE_CF_ACCESS_HEADERS = os.getenv("REQUIRE_CF_ACCESS_HEADERS", "0") == "1"

# APP CONFIG
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

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
USER_MSG_MAX_CHARS = int(os.getenv("USER_MSG_MAX_CHARS", "120000"))
SYSTEM_MAX_CHARS = int(os.getenv("SYSTEM_MAX_CHARS", "40000"))
ENFORCE_DIFF_FIRST = os.getenv("ENFORCE_DIFF_FIRST", "1") == "1"

# Caching modes
ENABLE_PREFIX_CACHE = os.getenv("ENABLE_PREFIX_CACHE", "1") == "1"
PREFIX_CACHE_TTL_SECONDS = int(os.getenv("PREFIX_CACHE_TTL_SECONDS", str(CACHE_TTL_SECONDS)))
ENABLE_TOOL_RESULT_DEDUP = os.getenv("ENABLE_TOOL_RESULT_DEDUP", "1") == "1"
