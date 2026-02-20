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
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
pgvector==0.2.4
prometheus-client==0.19.0
tenacity==8.2.3
```

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
