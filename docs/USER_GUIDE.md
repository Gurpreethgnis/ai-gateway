# AI Gateway User Guide

A complete guide for setting up, configuring, and using the AI Gateway across projects and devices.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Client Setup](#client-setup)
3. [Caching & Token Reduction](#caching--token-reduction)
4. [Multi-Project Setup](#multi-project-setup)
5. [Environment Variables Reference](#environment-variables-reference)
6. [Migrating to a New Device](#migrating-to-a-new-device)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
8. [Cost Optimization Tips](#cost-optimization-tips)

---

## Quick Start

### What You Need

| Component | Purpose | Required? |
|-----------|---------|-----------|
| Gateway URL | Your deployed gateway endpoint | Yes |
| Gateway API Key | Authentication for requests | Yes |
| Anthropic API Key | Upstream Claude access | Yes (server-side) |
| Redis | Caching, rate limiting | Recommended |
| PostgreSQL | Usage tracking, memory, projects | Optional |

### Your Gateway Details

After deployment, note your gateway URL:

```
Gateway URL: https://your-gateway.railway.app
Health Check: https://your-gateway.railway.app/health
```

### Verify It's Working

```bash
# Check health (use your API key)
curl https://your-gateway.railway.app/health \
  -H "Authorization: Bearer YOUR_GATEWAY_API_KEY"

# Expected: {"ok":true,"redis":true,...}
```

---

## Client Setup

### Cursor IDE

1. Open Cursor Settings (Ctrl+Shift+P → "Preferences: Open Settings (JSON)")
2. Add these settings:

```json
{
  "openai.baseUrl": "https://your-gateway.railway.app/v1",
  "openai.apiKey": "YOUR_GATEWAY_API_KEY"
}
```

Or via UI:
1. Settings → Models → OpenAI API Key
2. Enter your gateway API key
3. Settings → Models → Override OpenAI Base URL
4. Enter: `https://your-gateway.railway.app/v1`

### Continue Extension

Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Claude via Gateway",
      "provider": "openai",
      "model": "claude-sonnet-4-0",
      "apiBase": "https://your-gateway.railway.app/v1",
      "apiKey": "YOUR_GATEWAY_API_KEY"
    },
    {
      "title": "Claude Opus via Gateway",
      "provider": "openai", 
      "model": "claude-opus-4-5",
      "apiBase": "https://your-gateway.railway.app/v1",
      "apiKey": "YOUR_GATEWAY_API_KEY"
    }
  ]
}
```

### OpenAI SDK (JavaScript/TypeScript)

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://your-gateway.railway.app/v1',
  apiKey: 'YOUR_GATEWAY_API_KEY',
});

const response = await client.chat.completions.create({
  model: 'claude-sonnet-4-0',  // or 'sonnet', 'opus', 'claude-opus-4-5'
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});

for await (const chunk of response) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-gateway.railway.app/v1",
    api_key="YOUR_GATEWAY_API_KEY",
)

response = client.chat.completions.create(
    model="claude-sonnet-4-0",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### cURL

```bash
curl https://your-gateway.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_GATEWAY_API_KEY" \
  -d '{
    "model": "claude-sonnet-4-0",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Choosing the model in Cursor & Continue

In the model dropdown or settings, use the **model name** (no prefix required):

| What you type | Gateway sends to Anthropic |
|---------------|----------------------------|
| `sonnet` | claude-sonnet-4-0 |
| `opus` | claude-opus-4-5 |
| `claude-sonnet-4-0` | claude-sonnet-4-0 |
| `claude-opus-4-5` | claude-opus-4-5 |

- You do **not** need `MODEL:Sonnet` or any prefix—just **`sonnet`** or **`claude-sonnet-4-0`**.
- **Sonnet** is faster and cheaper; use it for most coding. **Opus** is for harder tasks (architecture, security, refactors).

### Using new models (e.g. Claude 4.6)

When Anthropic releases a new model (e.g. `claude-sonnet-4-6`), use the **full model ID** in your IDE:

- In Cursor/Continue: set model to **`claude-sonnet-4-6`** (or whatever ID Anthropic publishes).
- The gateway passes through any `claude-*` model ID, so new models work without a gateway update.

### Changing default models (server / deployer)

If you run the gateway, you can change which Claude model is used for the `sonnet` and `opus` aliases:

```bash
DEFAULT_MODEL=claude-sonnet-4-0    # Used when user picks "sonnet"
OPUS_MODEL=claude-opus-4-5        # Used when user picks "opus"
```

Set these in Railway (or your host) environment variables. Users still choose **`sonnet`** or **`opus`** in the IDE; the gateway maps them to your configured IDs.

### Available model aliases (reference)

| Model Alias | Actual Model | Use Case |
|-------------|--------------|----------|
| `sonnet` | claude-sonnet-4-0 | General coding |
| `opus` | claude-opus-4-5 | Complex architecture |
| `claude-sonnet-4-0` | claude-sonnet-4-0 | Explicit selection |
| `claude-opus-4-5` | claude-opus-4-5 | Explicit selection |
| `MYMODEL:sonnet` | claude-sonnet-4-0 | Prefixed (if MODEL_PREFIX set) |
| `MYMODEL:opus` | claude-opus-4-5 | Prefixed (if MODEL_PREFIX set) |

---

## Caching & Token Reduction

The gateway implements multiple caching layers to reduce token usage by 70-80%.

### How Caching Works

```
Request Flow:
┌─────────────┐
│   Client    │
└──────┬──────┘
       ▼
┌──────────────────────────────────────┐
│         Response Cache (Redis)       │ ← Identical requests return cached
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│     IDE Boilerplate Stripping        │ ← Removes repeated instructions
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│      File Hash Deduplication         │ ← Replaces duplicate files with refs
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│    Anthropic Prompt Caching          │ ← System prompts cached server-side
└──────┬───────────────────────────────┘
       ▼
┌─────────────┐
│  Anthropic  │
└─────────────┘
```

### Verify Caching is Working

Check response headers:

```bash
# First request - should be MISS
curl -I https://your-gateway.railway.app/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"Say hello"}]}'

# Look for: X-Cache: MISS

# Same request again - should be HIT
# Look for: X-Cache: HIT
```

### Cache Configuration (Server-Side)

These are configured on the Railway deployment:

```bash
# Response caching (30 min default)
CACHE_TTL_SECONDS=1800

# File hash caching (1 hour default)
ENABLE_FILE_HASH_CACHE=1
FILE_HASH_CACHE_TTL=3600

# Anthropic native prompt caching
ENABLE_ANTHROPIC_CACHE_CONTROL=1

# IDE boilerplate removal
STRIP_IDE_BOILERPLATE=1

# Diff-first policy
ENFORCE_DIFF_FIRST=1

# Context pruning (emergency overflow protection)
ENABLE_CONTEXT_PRUNING=1
CONTEXT_MAX_TOKENS=80000
```

### Token Reduction Breakdown

| Layer | Savings | How It Works |
|-------|---------|--------------|
| Response Cache | 100% on repeats | Identical requests return cached response |
| IDE Boilerplate | 20-30% | Strips repeated Cursor/Continue instructions |
| File Hash Dedup | 30-50% | Replaces unchanged files with hash references |
| Anthropic Caching | 80-90% on system | System prompts cached at Anthropic (90% cheaper) |
| Diff-First | 50-70% on edits | Model returns diffs instead of full files |
| Context Pruning | Prevents overflow | Summarizes old messages when context too large |

---

## Multi-Project Setup

For teams or multiple projects with different configurations.

### Enable Multi-Project Mode

Set on Railway:

```bash
ENABLE_MULTI_PROJECT=1
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
```

### Create Projects via Admin API

```bash
# Set your admin key
ADMIN_KEY="your-admin-api-key"

# Create a new project
curl -X POST https://your-gateway.railway.app/admin/projects \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "frontend-team",
    "rate_limit_rpm": 100,
    "config": {
      "default_model": "claude-sonnet-4-0",
      "max_tokens": 2000
    }
  }'

# Response includes the project's API key
# {"id": 1, "name": "frontend-team", "api_key": "generated-key-here", ...}
```

### List Projects

```bash
curl https://your-gateway.railway.app/admin/projects \
  -H "X-API-Key: $ADMIN_KEY"
```

### Update Project

```bash
curl -X PUT https://your-gateway.railway.app/admin/projects/1 \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "rate_limit_rpm": 200,
    "config": {
      "default_model": "claude-opus-4-5"
    }
  }'
```

### Delete Project

```bash
curl -X DELETE https://your-gateway.railway.app/admin/projects/1 \
  -H "X-API-Key: $ADMIN_KEY"
```

### Per-Project Configuration Options

```json
{
  "name": "project-name",
  "rate_limit_rpm": 100,
  "config": {
    "default_model": "claude-sonnet-4-0",
    "max_tokens": 2000,
    "temperature": 0.2,
    "system_prompt_prefix": "You are helping with the frontend project...",
    "allowed_tools": ["read_file", "write_file", "search"],
    "blocked_tools": ["dangerous_tool"]
  }
}
```

---

## Environment Variables Reference

### Required (Server-Side on Railway)

```bash
ANTHROPIC_API_KEY=sk-ant-...        # Your Anthropic API key
GATEWAY_API_KEY=your-gateway-key    # Key clients use to authenticate
```

### Infrastructure

```bash
REDIS_URL=redis://...               # Redis for caching (Railway provides this)
DATABASE_URL=postgresql+asyncpg://  # PostgreSQL for full features
```

### Token Reduction

```bash
STRIP_IDE_BOILERPLATE=1             # Remove IDE instructions
ENFORCE_DIFF_FIRST=1                # Prefer diffs over full files
ENABLE_ANTHROPIC_CACHE_CONTROL=1    # Use Anthropic's prompt caching
ENABLE_FILE_HASH_CACHE=1            # Deduplicate file content
FILE_HASH_CACHE_TTL=3600            # File cache TTL (seconds)
CACHE_TTL_SECONDS=1800              # Response cache TTL (seconds)
ENABLE_CONTEXT_PRUNING=1            # Prune long conversations
CONTEXT_MAX_TOKENS=80000            # Max tokens before pruning
```

### Truncation Limits

```bash
SYSTEM_MAX_CHARS=40000              # Max system prompt size
USER_MSG_MAX_CHARS=120000           # Max user message size
TOOL_RESULT_MAX_CHARS=20000         # Max tool result size
```

### Security

```bash
ADMIN_API_KEY=...                   # Admin endpoint access
ORIGIN_SECRET=...                   # Cloudflare origin verification
REQUIRE_CF_ACCESS_HEADERS=0         # Require CF Access headers
```

### Model Configuration

```bash
DEFAULT_MODEL=claude-sonnet-4-0     # Default model
OPUS_MODEL=claude-opus-4-5          # Opus model name
MODEL_PREFIX=MYMODEL:               # Optional model prefix
UPSTREAM_TIMEOUT_SECONDS=30         # API timeout
```

### Reliability

```bash
CIRCUIT_BREAKER_ENABLED=1           # Enable circuit breaker
CIRCUIT_BREAKER_THRESHOLD=5         # Failures before opening
CIRCUIT_BREAKER_TIMEOUT=60          # Seconds before retry
RETRY_ENABLED=1                     # Enable automatic retries
RETRY_MAX_ATTEMPTS=3                # Max retry attempts
RATE_LIMIT_ENABLED=1                # Enable rate limiting
RATE_LIMIT_RPM=60                   # Requests per minute
```

### Advanced Features

```bash
ENABLE_SMART_ROUTING=1              # Auto-route to Opus for complex tasks
OPUS_ROUTING_THRESHOLD=0.5          # Complexity threshold
ENABLE_MEMORY_LAYER=0               # Embedding-based memory
EMBEDDING_API_KEY=...               # OpenAI key for embeddings
ENABLE_MULTI_PROJECT=0              # Multi-project support
ENABLE_BATCH_API=1                  # Batch processing API
PROMETHEUS_ENABLED=1                # Prometheus metrics
```

---

## Migrating to a New Device

### What You Need to Save

Create a secure backup of these credentials:

```
┌─────────────────────────────────────────────┐
│           SAVE THESE SECURELY               │
├─────────────────────────────────────────────┤
│ Gateway URL:     https://your-gateway.railway.app │
│ Gateway API Key: YOUR_GATEWAY_API_KEY       │
│ Admin API Key:   YOUR_ADMIN_API_KEY         │
└─────────────────────────────────────────────┘
```

### Setup on New Device

#### 1. Install Tools

```bash
# Install Cursor
# Download from https://cursor.sh

# Or install Continue extension in VS Code
code --install-extension continue.continue
```

#### 2. Configure Environment Variables (Optional)

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, or PowerShell profile):

```bash
# Bash/Zsh
export GW_KEY="your-gateway-api-key"
export GW_URL="https://your-gateway.railway.app"
export ADMIN_KEY="your-admin-api-key"

# PowerShell (add to $PROFILE)
$env:GW_KEY = "your-gateway-api-key"
$env:GW_URL = "https://your-gateway.railway.app"
$env:ADMIN_KEY = "your-admin-api-key"
```

#### 3. Configure Cursor

Settings JSON:

```json
{
  "openai.baseUrl": "https://your-gateway.railway.app/v1",
  "openai.apiKey": "YOUR_GATEWAY_API_KEY"
}
```

#### 4. Verify Connection

```bash
# Test health
curl $GW_URL/health -H "Authorization: Bearer $GW_KEY"

# Test chat
curl $GW_URL/v1/chat/completions \
  -H "Authorization: Bearer $GW_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"Hello!"}]}'
```

### Quick Setup Script

Save this as `setup-gateway.sh`:

```bash
#!/bin/bash

# AI Gateway Quick Setup
GW_URL="https://your-gateway.railway.app"

echo "Testing gateway connection..."
curl -s "$GW_URL/health" -H "Authorization: Bearer $1" | python3 -m json.tool

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Gateway connected successfully!"
    echo ""
    echo "Add to your shell profile:"
    echo "  export GW_KEY=\"$1\""
    echo "  export GW_URL=\"$GW_URL\""
    echo ""
    echo "Configure Cursor:"
    echo "  Base URL: $GW_URL/v1"
    echo "  API Key: $1"
else
    echo "❌ Connection failed. Check your API key."
fi
```

Usage:
```bash
chmod +x setup-gateway.sh
./setup-gateway.sh YOUR_API_KEY
```

---

## Monitoring & Troubleshooting

### Health Check

```bash
curl $GW_URL/health -H "Authorization: Bearer $GW_KEY"
```

Expected response:
```json
{
  "ok": true,
  "redis": true,
  "default_model": "claude-sonnet-4-0",
  "enable_anthropic_cache_control": true,
  "strip_ide_boilerplate": true,
  "enforce_diff_first": true
}
```

### Check Usage

```bash
# Overall usage
curl $GW_URL/admin/usage -H "X-API-Key: $ADMIN_KEY"

# Daily breakdown
curl $GW_URL/admin/usage/daily -H "X-API-Key: $ADMIN_KEY"

# Costs
curl $GW_URL/admin/costs -H "X-API-Key: $ADMIN_KEY"
```

### Check Errors

```bash
# Recent errors
curl $GW_URL/admin/errors -H "X-API-Key: $ADMIN_KEY"

# Error statistics
curl $GW_URL/admin/errors/stats -H "X-API-Key: $ADMIN_KEY"
```

### Prometheus Metrics

```bash
curl $GW_URL/admin/metrics -H "X-API-Key: $ADMIN_KEY"
```

Key metrics:
- `gateway_requests_total` - Total requests
- `gateway_tokens_total` - Tokens used
- `gateway_cache_hits_total` - Cache hits
- `gateway_cost_usd_total` - Cost in USD

### Common Issues

#### "Redis: false" in health check

Redis not connected. Check Railway:
1. Redis service is running
2. `REDIS_URL` environment variable is set
3. Redeploy the gateway service

#### Requests timing out

Increase timeout:
```bash
UPSTREAM_TIMEOUT_SECONDS=60
```

#### Rate limited

Check your rate limit:
```bash
curl -I $GW_URL/v1/chat/completions ...
# Look for X-RateLimit-Remaining header
```

Increase limit:
```bash
RATE_LIMIT_RPM=120
```

#### Cache not working

1. Verify Redis is connected (health check)
2. Check `CACHE_TTL_SECONDS` > 0
3. Response caching only works for non-streaming, text-only responses

#### Model not found errors

The gateway maps model names. Use:
- `sonnet` or `claude-sonnet-4-0`
- `opus` or `claude-opus-4-5`

Avoid made-up model names.

---

## Cost Optimization Tips

### 1. Enable All Caching Layers

Ensure these are all enabled:
```bash
ENABLE_ANTHROPIC_CACHE_CONTROL=1
ENABLE_FILE_HASH_CACHE=1
STRIP_IDE_BOILERPLATE=1
ENFORCE_DIFF_FIRST=1
```

### 2. Use Appropriate Truncation

For cost-sensitive projects:
```bash
SYSTEM_MAX_CHARS=30000
USER_MSG_MAX_CHARS=80000
TOOL_RESULT_MAX_CHARS=10000
```

### 3. Use Sonnet for Most Tasks

Sonnet is ~5x cheaper than Opus. Only use Opus for:
- Architecture decisions
- Security reviews
- Complex refactoring
- Production incident analysis

Smart routing handles this automatically:
```bash
ENABLE_SMART_ROUTING=1
OPUS_ROUTING_THRESHOLD=0.5
```

### 4. Monitor Usage Regularly

```bash
# Check daily costs
curl $GW_URL/admin/costs -H "X-API-Key: $ADMIN_KEY"
```

### 5. Set Project Rate Limits

Prevent runaway usage:
```bash
RATE_LIMIT_ENABLED=1
RATE_LIMIT_RPM=60
```

### 6. Use Context Pruning for Long Sessions

```bash
ENABLE_CONTEXT_PRUNING=1
CONTEXT_MAX_TOKENS=80000
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    AI GATEWAY QUICK REF                     │
├─────────────────────────────────────────────────────────────┤
│ Gateway URL:    https://your-gateway.railway.app              │
│ API Endpoint:   https://your-gateway.railway.app/v1           │
│                                                             │
│ ENDPOINTS:                                                  │
│   POST /v1/chat/completions  - Chat (OpenAI compatible)     │
│   GET  /v1/models            - List models                  │
│   GET  /health               - Health check                 │
│   GET  /admin/usage          - Usage stats                  │
│   GET  /admin/costs          - Cost stats                   │
│   GET  /admin/metrics        - Prometheus metrics           │
│                                                             │
│ MODELS:                                                     │
│   sonnet / claude-sonnet-4-0  - Fast, cheap ($3/M tokens)   │
│   opus / claude-opus-4-5      - Smart, expensive ($15/M)    │
│                                                             │
│ HEADERS:                                                    │
│   Authorization: Bearer YOUR_KEY                            │
│   Content-Type: application/json                            │
│                                                             │
│ RESPONSE HEADERS:                                           │
│   X-Cache: HIT/MISS          - Cache status                 │
│   X-RateLimit-Remaining: N   - Requests left                │
│   X-Gateway: claude-gateway  - Confirms gateway used        │
└─────────────────────────────────────────────────────────────┘
```

---

## Support

- **Repository**: https://github.com/Gurpreethgnis/ai-gateway
- **Health Check**: https://your-gateway.railway.app/health
- **Railway Dashboard**: https://railway.app (check deployment logs)
