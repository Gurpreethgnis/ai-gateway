"""
Ollama Provider - Local LLM integration with streaming support.

Supports:
- Local Ollama instances
- Cloudflare Access tunnel
- Model discovery
- Streaming responses
"""

import json
from typing import List, Dict, Any, AsyncIterator, Optional

import httpx

from gateway.providers.base import BaseProvider, CompletionResponse, StreamChunk
from gateway.logging_setup import log
from gateway import config


class OllamaProvider(BaseProvider):
    """Ollama provider for local LLM inference."""
    
    def __init__(self):
        self.base_url = getattr(config, "OLLAMA_URL", None) or getattr(config, "LOCAL_LLM_BASE_URL", None)
        if not self.base_url:
            raise ValueError("OLLAMA_URL or LOCAL_LLM_BASE_URL not configured")
        
        # Cloudflare Access support (optional)
        self.cf_client_id = getattr(config, "LOCAL_CF_ACCESS_CLIENT_ID", None)
        self.cf_client_secret = getattr(config, "LOCAL_CF_ACCESS_CLIENT_SECRET", None)
        
        self.timeout = getattr(config, "LOCAL_LLM_TIMEOUT_SECONDS", 120)
        self._discovered_models: List[str] = []
    
    @property
    def name(self) -> str:
        return "ollama"
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers including CF Access if configured."""
        headers = {"Content-Type": "application/json"}
        
        if self.cf_client_id and self.cf_client_secret:
            headers["CF-Access-Client-Id"] = self.cf_client_id
            headers["CF-Access-Client-Secret"] = self.cf_client_secret
        
        return headers
    
    def supports_tools(self) -> bool:
        # Ollama doesn't natively support OpenAI-style tool calling
        # Some models have limited function calling but not reliable
        return False
    
    def supports_vision(self) -> bool:
        # Some Ollama models support vision (llava, etc.)
        # but we conservatively return False for routing purposes
        return False
    
    def get_models(self) -> List[str]:
        """Return discovered models with ollama/ prefix."""
        return [f"ollama/{m}" for m in self._discovered_models]
    
    async def discover_models(self) -> List[str]:
        """Discover available models from Ollama server."""
        url = f"{self.base_url.rstrip('/')}/api/tags"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=self._build_headers())
                
                if resp.status_code == 200:
                    data = resp.json()
                    self._discovered_models = [
                        m.get("name", "") for m in data.get("models", [])
                    ]
                    log.info("Discovered %d Ollama models", len(self._discovered_models))
                    return self._discovered_models
                else:
                    log.warning("Ollama model discovery failed: %d", resp.status_code)
                    return []
        except Exception as e:
            log.warning("Ollama model discovery error: %r", e)
            return []
    
    async def pull_model(self, model: str) -> AsyncIterator[Dict]:
        """
        Pull a model from Ollama registry.
        
        Yields progress updates as dicts with 'status' and optionally
        'completed', 'total' for download progress.
        """
        url = f"{self.base_url.rstrip('/')}/api/pull"
        
        payload = {"name": model, "stream": True}
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:  # No timeout for large downloads
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self._build_headers(),
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                yield data
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            log.error("Ollama pull error for %s: %r", model, e)
            yield {"status": "error", "error": str(e)}
    
    async def delete_model(self, model: str) -> bool:
        """Delete a model from Ollama."""
        url = f"{self.base_url.rstrip('/')}/api/delete"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(
                    url,
                    json={"name": model},
                    headers=self._build_headers(),
                )
                return resp.status_code == 200
        except Exception as e:
            log.error("Ollama delete error for %s: %r", model, e)
            return False
    
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> CompletionResponse:
        """Non-streaming completion using Ollama API."""
        
        model_id = self.normalize_model_id(model)
        
        # Build messages with system prompt
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(self.normalize_messages(messages))
        
        payload = {
            "model": model_id,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        url = f"{self.base_url.rstrip('/')}/api/chat"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers=self._build_headers(),
                )
                
                if resp.status_code != 200:
                    error_text = resp.text[:500]
                    log.error("Ollama error %d: %s", resp.status_code, error_text)
                    raise Exception(f"Ollama error {resp.status_code}: {error_text}")
                
                data = resp.json()
                
                content = data.get("message", {}).get("content", "")
                
                # Ollama returns eval/prompt token counts
                eval_count = data.get("eval_count", 0)
                prompt_eval_count = data.get("prompt_eval_count", 0)
                
                return CompletionResponse(
                    content=content,
                    model=model_id,
                    input_tokens=prompt_eval_count,
                    output_tokens=eval_count,
                    finish_reason="stop",
                )
                
        except Exception as e:
            log.error("Ollama API error: %r", e)
            raise
    
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming completion using Ollama API."""
        
        model_id = self.normalize_model_id(model)
        
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(self.normalize_messages(messages))
        
        payload = {
            "model": model_id,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        url = f"{self.base_url.rstrip('/')}/api/chat"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self._build_headers(),
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        log.error("Ollama stream error %d: %s", response.status_code, error_text[:500])
                        raise Exception(f"Ollama error {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        
                        # Extract content from message
                        message = data.get("message", {})
                        content = message.get("content", "")
                        
                        # Check if done
                        done = data.get("done", False)
                        
                        if content:
                            yield StreamChunk(
                                content=content,
                                is_final=done,
                            )
                        
                        if done:
                            yield StreamChunk(
                                finish_reason="stop",
                                is_final=True,
                            )
                            break
                            
        except Exception as e:
            log.error("Ollama streaming error: %r", e)
            raise
    
    def normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Normalize messages to Ollama format (similar to OpenAI)."""
        normalized = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Skip tool-related messages (Ollama doesn't support)
            if role == "tool":
                continue
            
            # Handle content blocks - flatten to text
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            # Include tool results as text
                            text_parts.append(f"[Tool result]: {block.get('content', '')}")
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            
            # Skip empty messages
            if not content:
                continue
            
            normalized.append({
                "role": role if role in ["user", "assistant", "system"] else "user",
                "content": content,
            })
        
        return normalized
