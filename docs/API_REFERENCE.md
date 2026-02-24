# AI Gateway API Reference

Complete API documentation for the AI Gateway.

---

## Table of Contents

1. [Authentication](#authentication)
2. [Chat Completions](#chat-completions)
3. [Models](#models)
4. [Health & Status](#health--status)
5. [Admin Endpoints](#admin-endpoints)
6. [Batch Processing](#batch-processing)
7. [Projects](#projects)
8. [Error Handling](#error-handling)
9. [Headers](#headers)

---

## Authentication

All API requests require authentication via the `Authorization` header.

### Bearer Token

```bash
Authorization: Bearer YOUR_GATEWAY_API_KEY
```

### X-API-Key (Admin endpoints)

```bash
X-API-Key: YOUR_ADMIN_API_KEY
```

---

## Chat Completions

### Create Chat Completion

`POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint supporting all providers.

#### Request Body

```json
{
  "model": "claude-sonnet-4-0",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 4096,
  "temperature": 0.7,
  "stream": true,
  "tools": [],
  "tool_choice": "auto"
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model ID (see [Models](#models)) |
| `messages` | array | Yes | - | Conversation messages |
| `max_tokens` | integer | No | 4096 | Maximum tokens in response |
| `temperature` | float | No | 0.7 | Sampling temperature (0-2) |
| `stream` | boolean | No | false | Enable streaming response |
| `tools` | array | No | [] | Tool definitions |
| `tool_choice` | string/object | No | "auto" | Tool selection mode |
| `provider` | string | No | - | Force specific provider |

#### Message Format

```json
{
  "role": "user|assistant|system|tool",
  "content": "Message text or structured content",
  "tool_calls": [],
  "tool_call_id": "for tool results"
}
```

#### Response (Non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1708894252,
  "model": "claude-sonnet-4-0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 12,
    "total_tokens": 37
  }
}
```

#### Response (Streaming)

Server-sent events format:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1708894252,"model":"claude-sonnet-4-0","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1708894252,"model":"claude-sonnet-4-0","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: [DONE]
```

#### Example: Basic Chat

```bash
curl https://your-gateway.railway.app/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### Example: Streaming

```bash
curl https://your-gateway.railway.app/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'
```

#### Example: With Tools

```bash
curl https://your-gateway.railway.app/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-0",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

---

## Models

### List Models

`GET /v1/models`

Returns available models across all configured providers.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-sonnet-4-0",
      "object": "model",
      "created": 1708894252,
      "owned_by": "anthropic",
      "capabilities": {
        "supports_tools": true,
        "supports_vision": true,
        "supports_streaming": true
      }
    },
    {
      "id": "gpt-4o",
      "object": "model",
      "created": 1708894252,
      "owned_by": "openai",
      "capabilities": {
        "supports_tools": true,
        "supports_vision": true,
        "supports_streaming": true
      }
    }
  ]
}
```

### Available Models

#### Anthropic (Claude)

| Model ID | Context | Description |
|----------|---------|-------------|
| `claude-opus-4-5` | 200K | Most capable, complex reasoning |
| `claude-sonnet-4-0` | 200K | Balanced speed and capability |
| `claude-3-5-haiku-20241022` | 200K | Fast, cost-effective |

#### OpenAI

| Model ID | Context | Description |
|----------|---------|-------------|
| `gpt-4o` | 128K | Latest GPT-4 Omni |
| `gpt-4o-mini` | 128K | Cost-effective GPT-4 |
| `o1` | 128K | Reasoning model |
| `o1-mini` | 128K | Smaller reasoning model |

#### Google Gemini

| Model ID | Context | Description |
|----------|---------|-------------|
| `gemini-1.5-pro` | 1M | Advanced multimodal |
| `gemini-1.5-flash` | 1M | Fast multimodal |
| `gemini-2.0-flash` | 1M | Latest flash model |

#### Groq

| Model ID | Context | Description |
|----------|---------|-------------|
| `groq/llama-3.3-70b-versatile` | 128K | Llama 3.3 70B |
| `groq/mixtral-8x7b-32768` | 32K | Mixtral MoE |
| `groq/llama-3.1-8b-instant` | 128K | Fast Llama |

#### Local (Ollama)

| Model ID | Description |
|----------|-------------|
| `local:qwen2.5-coder:14b-instruct` | Qwen coding model |
| `local:llama3.2:latest` | Llama 3.2 |
| `local:deepseek-coder:6.7b` | DeepSeek Coder |

### Model Aliases

| Alias | Resolves To |
|-------|-------------|
| `sonnet` | `claude-sonnet-4-0` |
| `opus` | `claude-opus-4-5` |
| `haiku` | `claude-3-5-haiku-20241022` |
| `auto` | Smart routing selection |

---

## Health & Status

### Liveness Probe

`GET /live`

Minimal health check that returns immediately when server is up.

```json
{"live": true}
```

### Health Check

`GET /health`

Full health check including dependencies.

```json
{
  "ok": true,
  "redis": true,
  "database": true,
  "providers": ["anthropic", "openai", "groq"],
  "default_model": "claude-sonnet-4-0",
  "cache_enabled": true,
  "smart_routing_enabled": true
}
```

---

## Admin Endpoints

All admin endpoints require the `X-API-Key` header with `ADMIN_API_KEY`.

### Get Usage Statistics

`GET /admin/usage`

```json
{
  "total_requests": 15420,
  "total_tokens": {
    "input": 4532100,
    "output": 1245300
  },
  "by_provider": {
    "anthropic": {"requests": 12000, "tokens": 4200000},
    "openai": {"requests": 2500, "tokens": 1200000}
  }
}
```

### Get Daily Usage

`GET /admin/usage/daily`

```json
{
  "data": [
    {"date": "2026-02-23", "requests": 542, "tokens": 125000},
    {"date": "2026-02-22", "requests": 489, "tokens": 118000}
  ]
}
```

### Get Cost Statistics

`GET /admin/costs`

```json
{
  "total_cost_usd": 145.23,
  "by_provider": {
    "anthropic": 98.50,
    "openai": 46.73
  },
  "by_model": {
    "claude-sonnet-4-0": 65.20,
    "claude-opus-4-5": 33.30
  }
}
```

### Get Prometheus Metrics

`GET /admin/metrics`

Returns Prometheus-formatted metrics:

```
# HELP gateway_requests_total Total requests processed
# TYPE gateway_requests_total counter
gateway_requests_total{provider="anthropic",model="claude-sonnet-4-0",status="success"} 12543

# HELP gateway_tokens_total Total tokens processed
# TYPE gateway_tokens_total counter
gateway_tokens_total{provider="anthropic",direction="input"} 4532100

# HELP gateway_latency_seconds Request latency
# TYPE gateway_latency_seconds histogram
gateway_latency_seconds_bucket{le="0.1"} 5420
gateway_latency_seconds_bucket{le="0.5"} 10234

# HELP gateway_cache_hits_total Cache hit count
# TYPE gateway_cache_hits_total counter
gateway_cache_hits_total 3421
```

### Get Error Statistics

`GET /admin/errors`

```json
{
  "recent_errors": [
    {
      "timestamp": "2026-02-23T10:15:32Z",
      "error": "rate_limit_exceeded",
      "provider": "anthropic",
      "details": "Rate limit hit"
    }
  ],
  "error_counts": {
    "rate_limit_exceeded": 12,
    "provider_timeout": 3
  }
}
```

---

## Batch Processing

### Submit Batch Job

`POST /v1/batch/submit`

```json
{
  "requests": [
    {
      "custom_id": "req-1",
      "method": "POST",
      "url": "/v1/chat/completions",
      "body": {
        "model": "claude-sonnet-4-0",
        "messages": [{"role": "user", "content": "Hello"}]
      }
    },
    {
      "custom_id": "req-2",
      "method": "POST",
      "url": "/v1/chat/completions",
      "body": {
        "model": "claude-sonnet-4-0",
        "messages": [{"role": "user", "content": "World"}]
      }
    }
  ]
}
```

#### Response

```json
{
  "batch_id": "batch_abc123",
  "status": "processing",
  "total": 2,
  "completed": 0
}
```

### Get Batch Status

`GET /v1/batch/{batch_id}`

```json
{
  "batch_id": "batch_abc123",
  "status": "completed",
  "total": 2,
  "completed": 2,
  "failed": 0,
  "created_at": "2026-02-23T10:00:00Z",
  "completed_at": "2026-02-23T10:00:15Z"
}
```

### Get Batch Results

`GET /v1/batch/{batch_id}/results`

```json
{
  "results": [
    {
      "custom_id": "req-1",
      "status": "success",
      "response": {...}
    }
  ]
}
```

---

## Projects

Multi-project support for team/tenant isolation.

### Create Project

`POST /admin/projects`

```json
{
  "name": "frontend-team",
  "rate_limit_rpm": 100,
  "config": {
    "default_model": "claude-sonnet-4-0",
    "max_tokens": 2000
  }
}
```

#### Response

```json
{
  "id": 1,
  "name": "frontend-team",
  "api_key": "proj_abc123xyz",
  "rate_limit_rpm": 100,
  "config": {...},
  "created_at": "2026-02-23T10:00:00Z"
}
```

### List Projects

`GET /admin/projects`

### Update Project

`PUT /admin/projects/{id}`

### Delete Project

`DELETE /admin/projects/{id}`

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "model_not_found",
    "message": "The model 'invalid-model' does not exist",
    "param": "model"
  }
}
```

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 400 | Bad Request | Invalid parameters, model not found |
| 401 | Unauthorized | Missing or invalid API key |
| 403 | Forbidden | Rate limit exceeded (project) |
| 404 | Not Found | Endpoint not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Error | Server error |
| 502 | Bad Gateway | Provider unreachable |
| 503 | Service Unavailable | Circuit breaker open |
| 504 | Gateway Timeout | Provider timeout |

### Error Types

| Type | Description |
|------|-------------|
| `authentication_error` | Invalid API key |
| `invalid_request_error` | Malformed request |
| `rate_limit_error` | Rate limit exceeded |
| `provider_error` | Upstream provider error |
| `timeout_error` | Request timeout |

---

## Headers

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | `Bearer YOUR_API_KEY` |
| `Content-Type` | Yes | `application/json` |
| `X-Gateway-Skill` | No | Skill ID for prompt template |
| `X-Gateway-Smart-Routing` | No | `1` to enable, `0` to disable |
| `X-Gateway-Provider` | No | Force specific provider |
| `X-Project-ID` | No | Project ID for multi-tenant |

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Gateway` | Confirms gateway processed request |
| `X-Provider` | Provider used (anthropic, openai, etc.) |
| `X-Model` | Actual model used |
| `X-Cache` | `HIT` or `MISS` |
| `X-RateLimit-Limit` | Requests per minute limit |
| `X-RateLimit-Remaining` | Requests remaining |
| `X-RateLimit-Reset` | Reset timestamp |
| `X-Smart-Route-Decision` | Routing decision (local/sonnet/opus) |
| `X-Smart-Route-Phase` | Routing phase (heuristic/llm_classifier) |
| `X-Request-ID` | Unique request identifier |

---

## SDK Examples

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-gateway.railway.app/v1",
    api_key="YOUR_GATEWAY_API_KEY",
)

# Streaming chat
response = client.chat.completions.create(
    model="claude-sonnet-4-0",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'https://your-gateway.railway.app/v1',
  apiKey: 'YOUR_GATEWAY_API_KEY',
});

const stream = await client.chat.completions.create({
  model: 'gpt-4o',
  messages: [{ role: 'user', content: 'Hello!' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

### cURL

```bash
curl https://your-gateway.railway.app/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-1.5-pro",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## Rate Limits

Default rate limits (configurable per project):

| Limit | Default | Description |
|-------|---------|-------------|
| Requests per minute | 60 | `RATE_LIMIT_RPM` |
| Concurrent requests | 10 | Per provider |
| Max tokens per request | 32000 | `DEFAULT_MAX_TOKENS` |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1708894320
```

---

## Versioning

The API follows OpenAI's versioning conventions. The current version is accessible at `/v1/`.

---

## Support

- **Repository:** https://github.com/Gurpreethgnis/ai-gateway
- **Issues:** https://github.com/Gurpreethgnis/ai-gateway/issues
- **User Guide:** [docs/USER_GUIDE.md](USER_GUIDE.md)
