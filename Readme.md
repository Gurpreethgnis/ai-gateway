# Claude Gateway

**OpenAI-compatible API proxy for Anthropic Claude with built-in caching, token reduction, and cost optimization.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI endpoints, works with Cursor, Continue, and any OpenAI SDK
- **70-80% Token Reduction** - Multi-layer caching including Anthropic prompt caching, response caching, and file deduplication
- **Smart Model Routing** - Automatically routes complex tasks to Opus, simple tasks to Sonnet
- **Full Tool Support** - Translates OpenAI tool calls to Anthropic format with streaming support
- **Production Ready** - Circuit breakers, retries, rate limiting, and Prometheus metrics
- **Multi-Project Support** - Isolated configurations, rate limits, and usage tracking per project

---

## Quick Start

### 1. Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

### 2. Set Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...    # Required
GATEWAY_API_KEY=your-secret     # Required - clients use this to authenticate
REDIS_URL=                      # Optional - enables caching (Railway provides this)
```

### 3. Configure Your Client

**Cursor / Continue:**
```
Base URL: https://your-gateway.railway.app/v1
API Key: your-gateway-api-key
```

**OpenAI SDK:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-gateway.railway.app/v1",
    api_key="your-gateway-api-key",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-0",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Token Reduction

The gateway reduces token usage through multiple layers:

| Layer | Savings | Description |
|-------|---------|-------------|
| Response Cache | 100% on repeats | Identical requests return cached responses |
| Anthropic Prompt Cache | 80-90% on system | System prompts cached server-side |
| File Deduplication | 30-50% | Unchanged files replaced with references |
| IDE Boilerplate Strip | 20-30% | Removes repeated Cursor/Continue instructions |
| Diff-First Policy | 50-70% on edits | Model returns diffs instead of full files |

Enable all caching:
```bash
REDIS_URL=redis://...
ENABLE_ANTHROPIC_CACHE_CONTROL=1
ENABLE_FILE_HASH_CACHE=1
STRIP_IDE_BOILERPLATE=1
ENFORCE_DIFF_FIRST=1
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |
| `GET /admin/metrics` | Prometheus metrics |
| `GET /admin/usage` | Usage statistics |

---

## Environment Variables

### Required
```bash
ANTHROPIC_API_KEY=              # Anthropic API key
GATEWAY_API_KEY=                # Client authentication key
```

### Caching & Token Reduction
```bash
REDIS_URL=                      # Redis connection (enables caching)
CACHE_TTL_SECONDS=1800          # Response cache TTL
ENABLE_ANTHROPIC_CACHE_CONTROL=1
ENABLE_FILE_HASH_CACHE=1
STRIP_IDE_BOILERPLATE=1
ENFORCE_DIFF_FIRST=1
```

### Limits
```bash
SYSTEM_MAX_CHARS=40000
USER_MSG_MAX_CHARS=120000
TOOL_RESULT_MAX_CHARS=20000
DEFAULT_MAX_TOKENS=1200
```

### Reliability
```bash
CIRCUIT_BREAKER_ENABLED=1
RETRY_ENABLED=1
RATE_LIMIT_ENABLED=1
RATE_LIMIT_RPM=60
```

### Advanced
```bash
DATABASE_URL=                   # PostgreSQL (enables usage tracking, multi-project)
ENABLE_MULTI_PROJECT=0
ENABLE_SMART_ROUTING=1
PROMETHEUS_ENABLED=1
```

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for complete configuration reference.

---

## Architecture

```
Client (Cursor/Continue/SDK)
         │
         ▼
┌─────────────────────────────────┐
│        Claude Gateway           │
├─────────────────────────────────┤
│  Auth → Rate Limit → Cache      │
│         │                       │
│  Token Reduction Layer          │
│  • IDE boilerplate stripping    │
│  • File hash deduplication      │
│  • Context pruning              │
│         │                       │
│  Smart Routing (Sonnet/Opus)    │
│         │                       │
│  Reliability Layer              │
│  • Circuit breaker              │
│  • Exponential backoff          │
└─────────────────────────────────┘
         │
         ▼
   Anthropic Claude API
```

---

## Models

| Alias | Model | Best For |
|-------|-------|----------|
| `sonnet` | claude-sonnet-4-0 | General coding, fast responses |
| `opus` | claude-opus-4-5 | Architecture, complex reasoning |

Smart routing automatically selects Opus for complex tasks (architecture, security, migrations).

---

## Monitoring

### Prometheus Metrics

```bash
curl https://your-gateway/admin/metrics -H "X-API-Key: $ADMIN_KEY"
```

Key metrics:
- `gateway_requests_total` - Request count by model/status
- `gateway_tokens_total` - Token usage
- `gateway_cache_hits_total` - Cache hit rate
- `gateway_cost_usd_total` - Cost tracking

### Response Headers

```
X-Cache: HIT|MISS              # Cache status
X-RateLimit-Remaining: 59      # Rate limit status
X-Gateway: claude-gateway      # Confirms gateway routing
```

---

## Project Structure

```
├── app.py                 # FastAPI entrypoint
├── gateway/
│   ├── config.py          # Environment configuration
│   ├── cache.py           # Redis caching
│   ├── token_reduction.py # Truncation & boilerplate removal
│   ├── circuit_breaker.py # Reliability
│   ├── smart_routing.py   # Model selection
│   ├── routers/
│   │   ├── openai.py      # /v1/chat/completions
│   │   ├── admin.py       # /admin/* endpoints
│   │   └── health.py      # /health
│   └── ...
└── docs/
    └── USER_GUIDE.md      # Detailed usage guide
```

---

## Documentation

- [User Guide](docs/USER_GUIDE.md) - Complete setup and configuration guide
- [API Reference](#api-endpoints) - Available endpoints
- [Environment Variables](#environment-variables) - All configuration options

---

## License

MIT

---

## Keywords

`claude` `anthropic` `openai-compatible` `api-gateway` `llm-proxy` `ai-gateway` `cursor` `continue` `token-optimization` `prompt-caching` `fastapi` `python` `claude-sonnet` `claude-opus` `ai-infrastructure` `cost-optimization` `rate-limiting` `circuit-breaker`
