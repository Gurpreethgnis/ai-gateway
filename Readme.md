# AI Gateway

**Production-ready multi-provider AI gateway with OpenAI-compatible API, intelligent routing, multi-layer caching, and 60-80% cost reduction.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

A unified API gateway that routes requests to multiple AI providers through a single OpenAI-compatible interface. Use Claude, GPT-4o, Gemini, Groq, or local Ollama models—all from Cursor, Continue, or any OpenAI SDK client.

### Key Highlights

- **5 Providers** — Anthropic, OpenAI, Google Gemini, Groq, Ollama (local)
- **60-80% Cost Reduction** — Multi-layer caching, prompt optimization, smart routing
- **OpenAI-Compatible** — Drop-in replacement for any OpenAI SDK client
- **Smart Routing** — Auto-select cheap/fast models for simple tasks, powerful models for complex ones
- **Production Ready** — Circuit breakers, retries, rate limiting, Prometheus metrics

---

## Supported Providers

| Provider | Models | Features | Status |
|----------|--------|----------|--------|
| **Anthropic Claude** | Opus 4.5, Sonnet 4, Haiku | Streaming, tools, prompt caching | ✅ Full |
| **OpenAI** | GPT-4o, GPT-4o-mini, o1, o1-mini | Streaming, tools, vision | ✅ Full |
| **Google Gemini** | Gemini 1.5 Pro, 2.0 Flash | Streaming, tools, vision | ✅ Full |
| **Groq** | Llama 3.3 70B, Mixtral, Gemma | Streaming, tools | ✅ Full |
| **Local Ollama** | Qwen, Llama, DeepSeek, CodeLlama | Self-hosted, no API costs | ✅ Full |

---

## Quick Start

### 1. Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

### 2. Set Environment Variables

**Required:**
```bash
ANTHROPIC_API_KEY=sk-ant-...    # At least one provider key
GATEWAY_API_KEY=your-secret     # Client authentication
```

**Add more providers:**
```bash
OPENAI_API_KEY=sk-...           # Enable OpenAI GPT models
GEMINI_API_KEY=...              # Enable Google Gemini
GROQ_API_KEY=gsk_...            # Enable Groq fast inference
LOCAL_LLM_BASE_URL=https://...  # Enable local Ollama
REDIS_URL=redis://...           # Enables response caching
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

# Use any provider
response = client.chat.completions.create(
    model="claude-sonnet-4-0",  # or gpt-4o, gemini-1.5-pro, auto
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Documentation

📖 **[User Guide](docs/USER_GUIDE.md)** — Complete setup, configuration, and usage

📚 **[API Reference](docs/API_REFERENCE.md)** — Full API documentation

💰 **[Cost Savings](COST_SAVINGS_IMPLEMENTATION.md)** — Token reduction techniques

🧠 **[Smart Routing](SMART_ROUTING_V2_IMPLEMENTATION.md)** — Routing system details

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **OpenAI-Compatible API** | Drop-in replacement for any OpenAI SDK client |
| **5 Provider Support** | Anthropic, OpenAI, Gemini, Groq, Ollama |
| **Smart Model Routing** | 2-phase routing: fast heuristics + LLM classifier |
| **Cascade Routing** | FrugalGPT-style: try cheap first, escalate if needed |
| **Full Tool Support** | OpenAI ↔ Anthropic tool call translation |
| **Skills System** | Curated prompt templates for structured workflows |

### Cost Optimization (60-80% Reduction)

| Layer | Savings | Description |
|-------|---------|-------------|
| **Response Cache** | 100% on repeats | Identical requests return cached (Redis) |
| **Anthropic Prompt Cache** | 80-90% on system | System prompts at ~10% cost |
| **File Deduplication** | 30-50% | Unchanged files as hash refs |
| **IDE Boilerplate Strip** | 20-30% | Remove Cursor/Continue noise |
| **Diff-First Policy** | 50-70% on edits | Return diffs vs full files |

### Production Ready

- **Circuit Breakers** — Automatic failover on errors
- **Exponential Backoff** — Smart retry logic
- **Rate Limiting** — Per-project request limits
- **Prometheus Metrics** — Full observability
- **Multi-Project Support** — Isolated configs per project

---

## Smart Routing

The gateway automatically selects the best model based on request complexity:

```
Phase 1: Fast Heuristics (<1ms)
├─ Tools present?          → Claude (tools required)
├─ Context >30K chars?     → Claude (local too small)
├─ Simple question?        → Local (cheap/fast)
├─ Deep reasoning keywords?→ Opus (complex)
└─ Multiple files?         → Sonnet (code task)

Phase 2: LLM Classifier (ambiguous only)
└─ Uses local model to classify: local/sonnet/opus
```

**Usage:**
```bash
# Let gateway choose
model="auto"

# Or via header
X-Gateway-Smart-Routing: 1
```

---

## Skills System

Curated prompt templates for structured workflows:

| Skill ID | Purpose |
|----------|---------|
| `brainstorming` | Structured feature planning |
| `systematic-debugging` | Methodical bug analysis |
| `code-review` | Thorough review checklist |
| `architecture-review` | System design analysis |
| `security-audit` | Vulnerability scan |
| `quick-fix` | Minimal changes for simple issues |

**Usage:**
```bash
curl -X POST https://your-gateway/v1/chat/completions \
  -H "X-Gateway-Skill: systematic-debugging" \
  -d '{"messages": [{"role": "user", "content": "Bug description"}]}'
```

---

## Project Status

### ✅ Implemented

| Category | Features |
|----------|----------|
| **Providers** | Anthropic, OpenAI, Gemini, Groq, Ollama |
| **Caching** | Response, Prompt, Semantic, File Dedup |
| **Routing** | Smart (2-phase), Cascade, Skills |
| **Reliability** | Circuit Breaker, Retry, Rate Limit |
| **Observability** | Prometheus, Usage Tracking, Admin UI |
| **Multi-tenant** | Projects, Per-project Config/Limits |
| **Advanced** | Batch API, Memory, Plugins, Repo Map |
| **Dashboard** | Project selector, per-project routing preferences, live preference→routing wiring |

### Recent Fixes (March 2026)

| Fix | Detail |
|-----|--------|
| **Dashboard preferences now affect routing** | Saved cost/speed bias sliders are read from DB before every routing decision — no longer ignored |
| **Project selector in dashboard** | New dropdown lets you switch between projects; preferences and model toggles update per-project |
| **`GET /api/projects` endpoint** | Dashboard-safe projects list (no admin key needed) |
| **Fixed `api_key` column bug** | Dashboard API was querying a non-existent column; now correctly uses numeric project `id` |
| **Fixed stats endpoint** | `GET /api/stats` now queries `usage_records` instead of the non-existent `metrics` table |
| **Ollama CF Access fix** | Local Ollama URLs (localhost / private IPs) no longer require Cloudflare Access credentials |
| **Migration 003** | Adds missing `progress`, `error`, `completed_at` columns to `ollama_pull_jobs` |
| **Resolved merge conflicts** | Stale conflict markers removed from `openai.py` and `dashboard.py` |

### New Features (Round 2)

| Feature | Detail |
|---------|--------|
| **Embeddings API** | `POST /v1/embeddings` — dispatches to OpenAI, Gemini (`text-embedding-004`), or Ollama (`/api/embed`) providers; returns OpenAI-compatible response |
| **Audit Log UI** | JS-driven paginated table in dashboard under "Recent Requests" — model filter, cached-only toggle, CSV export, `GET /api/usage` backend |
| **Usage Quotas** | `monthly_spend_limit_usd` column on projects; enforced via `check_project_quota()` (HTTP 402) on every chat request; Redis 60s cache; `GET/PUT /api/projects/{id}/quota` endpoints |
| **Webhooks** | Per-project webhook table; `GET/POST /api/webhooks`, delete and test endpoints; HMAC-SHA256 signed delivery; wired into telemetry on `request.complete` and `error.upstream` events; global `WEBHOOK_URL` env-var fallback |
| **Expanded cost table** | `COST_PER_1K_TOKENS` now covers Anthropic, OpenAI (incl. embeddings), Gemini, Groq, and Ollama models |
| **Migrations 004 & 005** | SQL files for `monthly_spend_limit_usd` column and `webhooks` table |

### 🚧 Roadmap — Sorted by Doability

> Items marked ✨ are recommended next steps given the current codebase.

#### 🟢 Easy (days)

| Feature | Description |
|---------|-------------|
| ✅ **Embeddings API** | `POST /v1/embeddings` — OpenAI-compatible, delegates to OpenAI / Gemini / Ollama providers. |
| ✅ **Request Audit Log UI** | JS-driven paginated audit log in the dashboard with model/cached filters and CSV export. Served from `GET /api/usage`. |
| ✅ **Webhook Integrations** | Per-project webhook CRUD (`GET/POST /api/webhooks`, delete, test). Signed HMAC-SHA256 delivery. Global env-var fallback (`WEBHOOK_URL`). |
| ✅ **Usage Quotas / Billing** | Per-project monthly spend cap (`monthly_spend_limit_usd`). Enforced at request time (`POST /v1/chat/completions`), with 60s Redis cache. |

#### 🟡 Medium (1–2 weeks)

| Feature | Description |
|---------|-------------|
| **A/B Testing** | Split traffic between two models, compare latency/cost/quality in usage_records. Needs a split-config table and dashboard view. |
| **WebSocket Support** | Real-time bidirectional streaming endpoint. FastAPI supports it natively; need to adapt the streaming response path. |
| **MCP Server Support** | Model Context Protocol — expose gateway as an MCP server. Well-defined spec, medium integration effort. |
| **Audio / Speech** | `/v1/audio/transcriptions` (Whisper) and `/v1/audio/speech` (TTS). OpenAI and Groq both expose Whisper; straightforward provider addition. |
| **Image Generation** | `/v1/images/generations` — DALL-E 3 via OpenAI provider, Stable Diffusion via Ollama. New provider method + route. |
| **Learned Router (RouteLLM)** | Train a lightweight classifier on logged routing outcomes from `routing_trace`. Requires data collection plumbing first. |

#### 🔴 Hard (weeks+)

| Feature | Description |
|---------|-------------|
| **SSO / SAML Auth** | Enterprise auth. Needs an auth library (python-saml / authlib) and session management overhaul. |
| **RAG Integration** | Built-in vector store — pgvector already partially wired (`embedding_chunks` table), but chunking pipeline and retrieval API are missing. |
| **Agent Workflows** | Multi-step orchestration with tool loops, planning, and state management. Architectural addition. |

### Contributing

1. Fork the repository
2. Pick a feature from the roadmap above (start with the 🟢 Easy ones)
3. Create feature branch: `git checkout -b feature/your-feature`
4. Submit a pull request

---

## Architecture

```
├── app.py                     # FastAPI entrypoint
├── gateway/
│   ├── providers/             # Provider implementations
│   │   ├── anthropic_provider.py
│   │   ├── openai_provider.py
│   │   ├── gemini_provider.py
│   │   ├── groq_provider.py
│   │   └── ollama_provider.py
│   ├── smart_routing.py       # 2-phase routing
│   ├── cascade_router.py      # FrugalGPT cascade
│   ├── skills.py              # Prompt templates
│   ├── cache.py               # Redis caching
│   ├── semantic_cache.py      # Embedding cache
│   ├── circuit_breaker.py     # Reliability
│   └── routers/               # API endpoints
├── docs/                      # Documentation
└── migrations/                # SQL migrations
```

---

## Environment Variables

### Provider Keys (at least one required)

```bash
ANTHROPIC_API_KEY=          # Anthropic Claude
OPENAI_API_KEY=             # OpenAI GPT models
GEMINI_API_KEY=             # Google Gemini
GROQ_API_KEY=               # Groq fast inference
LOCAL_LLM_BASE_URL=         # Ollama endpoint
```

### Core

```bash
GATEWAY_API_KEY=            # Client authentication (required)
REDIS_URL=                  # Redis for caching
DATABASE_URL=               # PostgreSQL for features
```

### Cost Optimization

```bash
ENABLE_ANTHROPIC_CACHE_CONTROL=1
ENABLE_FILE_HASH_CACHE=1
STRIP_IDE_BOILERPLATE=1
ENFORCE_DIFF_FIRST=1
```

### Routing

```bash
ENABLE_SMART_ROUTING=1
SMART_ROUTING_MODE=local_first
ENABLE_CASCADE_ROUTING=0
```

### Reliability

```bash
CIRCUIT_BREAKER_ENABLED=1
RETRY_ENABLED=1
RATE_LIMIT_ENABLED=1
RATE_LIMIT_RPM=60
```

---

## License

MIT

---

## Links

- **Repository:** https://github.com/Gurpreethgnis/ai-gateway
- **User Guide:** [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- **API Reference:** [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Issues:** https://github.com/Gurpreethgnis/ai-gateway/issues

---

## Keywords

`ai-gateway` `llm-proxy` `openai-compatible` `anthropic` `claude` `gpt-4` `gemini` `groq` `ollama` `multi-provider` `cursor` `continue` `smart-routing` `cost-optimization` `fastapi` `python`
