# Coding Gateway

### OpenAI-Compatible Claude Infrastructure Layer

**Tool Calling • Streaming • Token Reduction • Secure Routing • Full Observability**

---

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-production%20ready-green)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-purple)
![OpenAI Compatible](https://img.shields.io/badge/OpenAI-compatible-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-production--grade-success)

---

## What This Is

Claude Gateway is a secure, production-grade AI infrastructure layer that:

* Proxies Anthropic Claude models
* Exposes fully **OpenAI-compatible APIs**
* Enables **Cursor / Continue agent mode**
* Translates tool-calling between OpenAI and Anthropic formats
* Implements aggressive **token reduction (70–80% target savings)**
* Provides **full observability** with Prometheus metrics and usage tracking
* Includes **reliability features** like circuit breakers, retries, and rate limiting
* Supports **advanced features** like embedding memory, batch processing, and plugin tools

It allows you to keep using your IDE — while moving intelligence and efficiency into your own infrastructure.

---

# Why This Exists

Modern AI IDE integrations (Cursor, Continue, etc.):

* Re-send massive instruction blocks every turn
* Re-send full file context repeatedly
* Provide no routing control
* Provide no caching layer
* Offer little cost observability

This gateway fixes that.

It turns Claude into infrastructure — not a subscription feature.

---

# Architecture

```
Cursor / Continue / OpenAI SDK
            ↓
    ┌───────────────────────────────────────┐
    │          FastAPI Gateway              │
    ├───────────────────────────────────────┤
    │  Rate Limiter → Auth → Smart Router   │
    │         ↓              ↓              │
    │  Circuit Breaker ← Retry Handler      │
    │         ↓                             │
    │  ┌─────────────────────────────┐      │
    │  │      Optimization Layer    │      │
    │  │  • File Hash Cache         │      │
    │  │  • Repo Map Generator      │      │
    │  │  • Context Pruner          │      │
    │  └─────────────────────────────┘      │
    │         ↓                             │
    │  ┌─────────────────────────────┐      │
    │  │      Memory Layer          │      │
    │  │  • Embedding Memory        │      │
    │  │  • Plugin Registry         │      │
    │  └─────────────────────────────┘      │
    └───────────────────────────────────────┘
            ↓                ↓
    ┌───────────┐    ┌───────────────┐
    │ Anthropic │    │ Embedding API │
    │    API    │    │  (OpenAI)     │
    └───────────┘    └───────────────┘
            ↓
    ┌─────────────────────────────────┐
    │          Storage Layer          │
    │  Redis │ PostgreSQL │ Prometheus│
    └─────────────────────────────────┘
```

## The Gateway Owns:

* Context control
* Tool-call translation
* Streaming translation
* Token reduction
* Model routing
* Caching
* Security
* Observability
* Rate limiting
* Reliability (circuit breaker, retries)
* Memory (embedding-based context)

The IDE becomes UI only.

---

# OpenAI-Compatible API

## Supported Endpoints

| Endpoint                    | Purpose                      |
| --------------------------- | ---------------------------- |
| `POST /v1/chat/completions` | OpenAI-compatible chat       |
| `POST /chat/completions`    | Alias                        |
| `GET /v1/models`            | Model list                   |
| `GET /models`               | Alias                        |
| `POST /v1/batch`            | Batch request processing     |
| `GET /v1/batch/{id}`        | Batch status                 |
| `GET /admin/metrics`        | Prometheus metrics           |
| `GET /admin/usage`          | Usage reporting              |
| `GET /admin/costs`          | Cost reporting               |
| `GET /admin/projects`       | Project management           |

Works with:

* OpenAI JS SDK
* Continue
* Cursor
* Any OpenAI-style client

---

# Tool Calling (OpenAI ⇄ Claude)

### OpenAI Request

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "parameters": { ... }
      }
    }
  ]
}
```

### Translated to Anthropic

```json
{
  "tools": [
    {
      "name": "read_file",
      "input_schema": { ... }
    }
  ]
}
```

### Anthropic → OpenAI tool_calls

```json
"tool_calls": [
  {
    "id": "tool_123",
    "type": "function",
    "function": {
      "name": "read_file",
      "arguments": "{...}"
    }
  }
]
```

Streaming tool deltas are emitted in proper OpenAI SSE format.

Agent mode works end-to-end.

---

# Streaming

Implements proper OpenAI streaming:

```
data: { "delta": {"content": "..."} }

data: { "delta": {"tool_calls": [...]} }

data: { "finish_reason": "stop" }

data: [DONE]
```

Supports:

* Text streaming
* Tool-call streaming
* Proper finish_reason semantics
* Usage data in final chunk (stream_options.include_usage)

---

# Token Reduction

The core optimization layer lives in the gateway.

### 1. IDE Boilerplate Stripping

Removes large repeated Continue/Cursor instruction blocks.

### 2. Hard Truncation Caps

Configurable caps on:

* System prompts (`SYSTEM_MAX_CHARS`)
* User messages (`USER_MSG_MAX_CHARS`)
* Tool results (`TOOL_RESULT_MAX_CHARS`)

### 3. Anthropic Native Prompt Caching

Uses Anthropic's native `cache_control` API for system prompts:

```json
{
  "system": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]
}
```

This enables server-side prompt caching without gateway-side complexity.

### 4. Diff-First Enforcement

Injects policy:

```
Respond with unified diffs unless full file explicitly requested.
```

Prevents large file dumps.

### 5. File-Hash Diff Caching

Detects repeated file content in tool results and replaces with compact references:

```
[FILE_REF:abc123def456] (unchanged from previous request)
```

### 6. Context Window Pruning

Intelligent message pruning when context exceeds limits:

* Keeps system messages and recent turns
* Summarizes dropped middle sections
* Preserves important messages (errors, requirements)

---

# Model Routing

## Basic Routing

| Task Type                                        | Model         |
| ------------------------------------------------ | ------------- |
| Normal coding                                    | Claude Sonnet |
| Architecture / Production / Security / Migration | Claude Opus   |

## Smart Routing (Enhanced)

Scoring-based model selection with:

* Keyword matching (architect, design, security, etc.)
* Complexity heuristics (message length, tool count, file references)
* Historical success rate learning
* Configurable threshold (`OPUS_ROUTING_THRESHOLD`)

---

# Security Model

Layered security:

## 1. Cloudflare Access

Protects public domain with service tokens.

## 2. Origin Lockdown

`X-Origin-Secret` required to prevent direct backend bypass.

## 3. Gateway API Key

Accepts:

* `X-API-Key`
* `api-key`
* `Authorization: Bearer`

Unauthorized requests are rejected.

## 4. Rate Limiting

Per-project sliding window rate limiting with Redis.

---

# Project Structure

```
├── app.py                    # FastAPI entrypoint with lifespan
├── Procfile                  # Railway deployment
├── requirements.txt
│
└── gateway/
    ├── __init__.py
    ├── config.py             # Environment config (all features)
    ├── logging_setup.py      # Logging + exception handlers
    ├── security.py           # API key + origin validation
    ├── routing.py            # Basic model selection
    ├── cache.py              # Redis caching helpers
    ├── db.py                 # SQLAlchemy models + async session
    │
    ├── anthropic_client.py   # Anthropic SDK wrapper
    ├── openai_tools.py       # OpenAI ↔ Anthropic tool translation
    ├── token_reduction.py    # Truncation + diff-first policy
    ├── models.py             # Pydantic request models
    │
    ├── circuit_breaker.py    # Distributed circuit breaker
    ├── retry.py              # Exponential backoff retries
    ├── telemetry.py          # Structured error events
    ├── metrics.py            # Prometheus metrics
    ├── rate_limit.py         # Sliding window rate limiter
    │
    ├── file_cache.py         # File hash deduplication
    ├── repo_map.py           # Repository structure tracking
    ├── context_pruner.py     # Intelligent context pruning
    ├── smart_routing.py      # Enhanced model selection
    │
    ├── memory.py             # Embedding-based memory
    ├── batch.py              # Background batch processing
    ├── projects.py           # Multi-project configurations
    ├── plugins.py            # Custom tool registry
    │
    └── routers/
        ├── __init__.py
        ├── openai.py         # /v1/chat/completions (integrated)
        ├── chat.py           # /chat (simple endpoint)
        ├── health.py         # /health + debug endpoints
        └── admin.py          # /admin/* (metrics, usage, projects)
```

---

# Environment Variables

## Required

```
ANTHROPIC_API_KEY=
GATEWAY_API_KEY=
```

## Core Configuration

```
REDIS_URL=                          # Redis for caching, rate limiting, circuit breaker
DATABASE_URL=                       # PostgreSQL for usage tracking, projects, memory

DEFAULT_MODEL=claude-sonnet-4-0
OPUS_MODEL=claude-opus-4-5
CACHE_TTL_SECONDS=1800
UPSTREAM_TIMEOUT_SECONDS=30
MODEL_PREFIX=MYMODEL:
```

## Security

```
ADMIN_API_KEY=                      # Admin endpoints (defaults to GATEWAY_API_KEY)
ORIGIN_SECRET=
REQUIRE_CF_ACCESS_HEADERS=1
```

## Token Reduction

```
STRIP_IDE_BOILERPLATE=1
ENFORCE_DIFF_FIRST=1
ENABLE_ANTHROPIC_CACHE_CONTROL=1

SYSTEM_MAX_CHARS=40000
USER_MSG_MAX_CHARS=120000
TOOL_RESULT_MAX_CHARS=20000
```

## Optimization Features

```
ENABLE_FILE_HASH_CACHE=1
FILE_HASH_CACHE_TTL=3600
ENABLE_REPO_MAP=0
ENABLE_CONTEXT_PRUNING=0
CONTEXT_MAX_TOKENS=100000
ENABLE_SMART_ROUTING=1
OPUS_ROUTING_THRESHOLD=0.5
```

## Observability

```
PROMETHEUS_ENABLED=1
```

## Rate Limiting

```
RATE_LIMIT_ENABLED=1
RATE_LIMIT_RPM=60
```

## Reliability

```
CIRCUIT_BREAKER_ENABLED=1
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60
RETRY_ENABLED=1
RETRY_MAX_ATTEMPTS=3
RETRY_BACKOFF_BASE=1.0
```

## Advanced Features

```
ENABLE_MEMORY_LAYER=0
EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
EMBEDDING_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small

ENABLE_BATCH_API=1
ENABLE_MULTI_PROJECT=0
ENABLE_PLUGIN_TOOLS=1
```

---

# Deployment

## Railway (Recommended)

1. Deploy FastAPI app
2. Add environment variables
3. Attach Redis
4. Attach PostgreSQL (for full features)
5. Configure domain
6. Enable Cloudflare Access
7. Set origin secret
8. Enforce HTTPS

## Dependencies

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
anthropic==0.34.2
httpx>=0.27,<0.28
redis==5.0.8
pydantic==2.8.2
sqlalchemy[asyncio]==2.0.36
asyncpg==0.31.0
pgvector==0.2.4
prometheus-client==0.19.0
tenacity==8.2.3
```

Note: SQLAlchemy 2.0.36+ and asyncpg 0.31.0+ are required for Python 3.13 compatibility.

---

# User Guide: Achieving 70-80% Token Reduction

This section explains how to configure the gateway to maximize token savings across your projects.

## Quick Start: Recommended Configuration

For maximum token savings, use this environment configuration:

```bash
# Core (Required)
ANTHROPIC_API_KEY=your-anthropic-key
GATEWAY_API_KEY=your-gateway-key

# Infrastructure (Recommended for full features)
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/gateway

# Token Reduction (All enabled for maximum savings)
STRIP_IDE_BOILERPLATE=1
ENFORCE_DIFF_FIRST=1
ENABLE_ANTHROPIC_CACHE_CONTROL=1
ENABLE_FILE_HASH_CACHE=1
FILE_HASH_CACHE_TTL=3600
ENABLE_CONTEXT_PRUNING=1
CONTEXT_MAX_TOKENS=80000

# Truncation Limits (tune based on your use case)
SYSTEM_MAX_CHARS=40000
USER_MSG_MAX_CHARS=100000
TOOL_RESULT_MAX_CHARS=15000

# Response Caching
CACHE_TTL_SECONDS=1800
```

## How Token Reduction Works

The gateway reduces tokens through multiple layers, each targeting different inefficiencies:

### Layer 1: IDE Boilerplate Stripping (20-30% savings)

IDEs like Cursor and Continue inject large instruction blocks with every request:

```
STRIP_IDE_BOILERPLATE=1
```

This automatically detects and removes:
- Repeated system instructions
- Formatting rules that appear in every request
- IDE-specific metadata

### Layer 2: Anthropic Native Prompt Caching (Up to 90% on system prompts)

```
ENABLE_ANTHROPIC_CACHE_CONTROL=1
```

This wraps your system prompt with Anthropic's native caching:

```json
{
  "system": [{
    "type": "text",
    "text": "Your system prompt...",
    "cache_control": {"type": "ephemeral"}
  }]
}
```

Benefits:
- Cached prompts cost only 10% of normal tokens
- 5-minute TTL on Anthropic's servers
- No Redis required for this feature

### Layer 3: File Hash Deduplication (30-50% savings on file-heavy workflows)

```
ENABLE_FILE_HASH_CACHE=1
FILE_HASH_CACHE_TTL=3600
```

When the same file content is sent multiple times:
1. First request: Full content sent, hash stored
2. Subsequent requests: Content replaced with reference

```
[FILE_REF:sha256:abc123...] (cached, see previous context)
```

Requires: Redis (primary) and optionally PostgreSQL (persistence)

### Layer 4: Response Caching (100% savings on repeat queries)

```
REDIS_URL=redis://localhost:6379
CACHE_TTL_SECONDS=1800
```

Identical requests return cached responses instantly:
- Cache key: SHA256 hash of request payload
- TTL: 30 minutes (configurable)
- Headers: `X-Cache: HIT` indicates cache hit

### Layer 5: Context Window Pruning (Emergency overflow protection)

```
ENABLE_CONTEXT_PRUNING=1
CONTEXT_MAX_TOKENS=80000
```

When conversation history exceeds limits:
1. System messages preserved
2. Recent 5 messages preserved
3. Important messages (errors, requirements) preserved
4. Middle messages summarized or dropped

### Layer 6: Diff-First Policy (50-70% savings on code edits)

```
ENFORCE_DIFF_FIRST=1
```

Injects instruction to return unified diffs instead of full files:

```
When editing code, respond with unified diffs unless full file explicitly requested.
```

---

## Configuration by Use Case

### Cursor / Continue (IDE Integration)

```bash
# Maximum savings for IDE workflows
STRIP_IDE_BOILERPLATE=1
ENFORCE_DIFF_FIRST=1
ENABLE_ANTHROPIC_CACHE_CONTROL=1
ENABLE_FILE_HASH_CACHE=1
ENABLE_CONTEXT_PRUNING=1
CONTEXT_MAX_TOKENS=80000

# Aggressive truncation for IDE messages
SYSTEM_MAX_CHARS=30000
USER_MSG_MAX_CHARS=80000
TOOL_RESULT_MAX_CHARS=10000
```

### API Integration (Backend Services)

```bash
# Balanced savings with full context preservation
STRIP_IDE_BOILERPLATE=0
ENFORCE_DIFF_FIRST=0
ENABLE_ANTHROPIC_CACHE_CONTROL=1
ENABLE_FILE_HASH_CACHE=0
ENABLE_CONTEXT_PRUNING=0

# Higher limits for API use
SYSTEM_MAX_CHARS=50000
USER_MSG_MAX_CHARS=150000
TOOL_RESULT_MAX_CHARS=30000

# Enable response caching
CACHE_TTL_SECONDS=3600
```

### Multi-Project Environment

```bash
# Enable per-project configuration
ENABLE_MULTI_PROJECT=1
DATABASE_URL=postgresql+asyncpg://...

# Rate limiting per project
RATE_LIMIT_ENABLED=1
RATE_LIMIT_RPM=100
```

Then configure per-project via Admin API:

```bash
# Create project with custom settings
curl -X POST https://your-gateway/admin/projects \
  -H "X-API-Key: $ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "frontend-team",
    "rate_limit_rpm": 200,
    "config": {
      "default_model": "claude-sonnet-4-0",
      "max_tokens": 2000,
      "allowed_tools": ["read_file", "write_file"]
    }
  }'
```

---

## Memory Layer (Long-term Context)

For workflows that benefit from persistent memory across conversations:

```bash
ENABLE_MEMORY_LAYER=1
DATABASE_URL=postgresql+asyncpg://...
EMBEDDING_API_KEY=your-openai-key  # or use OPENAI_API_KEY
EMBEDDING_MODEL=text-embedding-3-small
```

The memory layer:
1. Stores important assistant responses and user requirements
2. Retrieves relevant context via semantic similarity
3. Injects recalled memories into system prompts

This reduces the need to repeat context across sessions.

---

## Monitoring Token Savings

### Prometheus Metrics

```bash
PROMETHEUS_ENABLED=1
```

Key metrics for token monitoring:

| Metric | Description |
|--------|-------------|
| `gateway_tokens_total{type="input"}` | Total input tokens |
| `gateway_tokens_total{type="output"}` | Total output tokens |
| `gateway_cache_hits_total{cache_type="response"}` | Response cache hits |
| `gateway_cache_hits_total{cache_type="file_hash"}` | File hash cache hits |
| `gateway_cost_usd_total` | Total cost in USD |

### Admin Endpoints

```bash
# Usage statistics
curl https://your-gateway/admin/usage \
  -H "X-API-Key: $ADMIN_API_KEY"

# Cost breakdown
curl https://your-gateway/admin/costs \
  -H "X-API-Key: $ADMIN_API_KEY"

# Daily usage
curl https://your-gateway/admin/usage/daily \
  -H "X-API-Key: $ADMIN_API_KEY"
```

### Response Headers

Check these headers to verify savings:

```
X-Cache: HIT          # Response served from cache
X-Reduction: 1        # Token reduction was applied
X-Model-Source: custom # Request went through gateway
```

---

## Client Configuration Examples

### Cursor

In Cursor settings, configure OpenAI API:

```json
{
  "openai.baseUrl": "https://your-gateway-domain.com/v1",
  "openai.apiKey": "your-gateway-api-key"
}
```

### Continue

In `.continue/config.json`:

```json
{
  "models": [{
    "title": "Claude via Gateway",
    "provider": "openai",
    "model": "claude-sonnet-4-0",
    "apiBase": "https://your-gateway-domain.com/v1",
    "apiKey": "your-gateway-api-key"
  }]
}
```

### OpenAI SDK (JavaScript)

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://your-gateway-domain.com/v1',
  apiKey: 'your-gateway-api-key',
});

const response = await client.chat.completions.create({
  model: 'claude-sonnet-4-0',  // or 'sonnet', 'opus'
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-gateway-domain.com/v1",
    api_key="your-gateway-api-key",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-0",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
```

### cURL

```bash
curl https://your-gateway-domain.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-gateway-api-key" \
  -d '{
    "model": "claude-sonnet-4-0",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## Expected Savings Breakdown

| Feature | Typical Savings | Requirements |
|---------|-----------------|--------------|
| IDE Boilerplate Stripping | 20-30% | None |
| Anthropic Prompt Caching | 80-90% on system prompts | None |
| File Hash Deduplication | 30-50% on file operations | Redis |
| Response Caching | 100% on repeat queries | Redis |
| Context Pruning | Prevents overflows | None |
| Diff-First Policy | 50-70% on code edits | None |

**Combined Effect**: 70-80% overall token reduction in typical IDE workflows.

---

## Troubleshooting

### Caching Not Working

1. Check Redis connection:
```bash
curl https://your-gateway/health
# Should show redis_connected: true
```

2. Verify cache headers in responses:
```
X-Cache: MISS  # First request
X-Cache: HIT   # Subsequent identical requests
```

### High Token Usage Despite Configuration

1. Check if features are enabled:
```bash
curl https://your-gateway/health
# Review feature flags in response
```

2. Review Prometheus metrics:
```bash
curl https://your-gateway/admin/metrics | grep gateway_cache
```

3. Increase truncation aggressiveness:
```bash
TOOL_RESULT_MAX_CHARS=10000
USER_MSG_MAX_CHARS=60000
```

### Memory Layer Not Storing

1. Verify database connection
2. Check that `ENABLE_MEMORY_LAYER=1`
3. Ensure `EMBEDDING_API_KEY` is set

---

# Observability

## Prometheus Metrics

Available at `/admin/metrics`:

| Metric                           | Type      | Labels                          |
| -------------------------------- | --------- | ------------------------------- |
| `gateway_requests_total`         | Counter   | model, project, status, endpoint|
| `gateway_request_latency_seconds`| Histogram | model, project, endpoint        |
| `gateway_tokens_total`           | Counter   | model, project, type            |
| `gateway_cost_usd_total`         | Counter   | model, project                  |
| `gateway_active_requests`        | Gauge     | model                           |
| `gateway_circuit_state`          | Gauge     | upstream                        |
| `gateway_cache_hits_total`       | Counter   | cache_type                      |
| `gateway_rate_limit_hits_total`  | Counter   | project                         |
| `gateway_stream_duration_seconds`| Histogram | model, project                  |
| `gateway_upstream_errors_total`  | Counter   | model, error_type, status_code  |

## Admin Endpoints

| Endpoint                   | Description                    |
| -------------------------- | ------------------------------ |
| `GET /admin/metrics`       | Prometheus metrics             |
| `GET /admin/usage`         | Usage by model/project         |
| `GET /admin/usage/daily`   | Daily usage breakdown          |
| `GET /admin/costs`         | Cost summaries                 |
| `GET /admin/errors`        | Recent error events            |
| `GET /admin/errors/stats`  | Error statistics               |
| `GET /admin/projects`      | List projects                  |
| `POST /admin/projects`     | Create project                 |
| `PUT /admin/projects/{id}` | Update project                 |
| `DELETE /admin/projects/{id}` | Delete project              |

## Response Headers

```
X-Gateway: gursimanoor-gateway
X-Model-Source: custom
X-Cache: HIT|MISS
X-Reduction: 1
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 60
```

---

# Completed Features

### Infrastructure

* [x] FastAPI gateway deployed
* [x] Anthropic integration
* [x] Model routing (Sonnet/Opus)
* [x] OpenAI-compatible endpoints
* [x] Streaming support (SSE)
* [x] Database layer (PostgreSQL + SQLAlchemy async)

### Tool Calling

* [x] OpenAI → Anthropic tool translation
* [x] Anthropic → OpenAI tool_calls translation
* [x] Streaming tool-call delta support
* [x] Assistant tool history reconstruction
* [x] function_call legacy compatibility

### Token Reduction

* [x] IDE boilerplate stripping
* [x] Hard truncation caps
* [x] Anthropic native prompt caching (cache_control)
* [x] Diff-first enforcement
* [x] Text-only response caching (Redis)

### Security

* [x] Gateway API key enforcement
* [x] Cloudflare Access support
* [x] Origin secret enforcement
* [x] Request validation logging

### Optimization

* [x] File-hash diff caching
* [x] Repo map generation layer
* [x] Context window pruning heuristics
* [x] Intelligent Opus routing (smart routing with scoring)

### Observability

* [x] Prometheus metrics (requests, tokens, costs, latency)
* [x] Per-project usage tracking
* [x] Cost reporting endpoints
* [x] Rate limiting per API key/project

### Reliability

* [x] Exponential backoff retry mechanism
* [x] Circuit breaker for upstream failures
* [x] Structured error telemetry export

### Advanced Features

* [x] Embedding-backed memory layer
* [x] Background batch execution support
* [x] Multi-project configuration profiles
* [x] Plugin tool registry

---

# Design Philosophy

This project treats AI as infrastructure.

* IDE = UX
* Gateway = Policy Engine
* Claude = Compute Layer

You control:

* Context
* Cost
* Routing
* Security
* Behavior
* Reliability
* Memory

The IDE does not.

---

# License

MIT

---

# Contributions

Issues and pull requests welcome.

For major changes, open an issue first to discuss architecture impact.
