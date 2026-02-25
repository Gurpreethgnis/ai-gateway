"""
Anthropic Provider - Claude API integration.

Wraps the existing anthropic_client.py functionality with the BaseProvider interface.
"""

import asyncio
import json
from typing import List, Dict, Any, AsyncIterator, Optional

from gateway.canonical_format import CanonicalMessage, canonical_to_anthropic_messages, to_canonical_messages
from gateway.providers.base import BaseProvider, CompletionResponse, StreamChunk
from gateway.logging_setup import log
from gateway import config


class AnthropicProvider(BaseProvider):
    """Anthropic API provider for Claude models."""
    
    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        
        self._client = None
        self._async_client = None
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    @property
    def client(self):
        """Lazy-load synchronous Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    def supports_tools(self) -> bool:
        return True
    
    def supports_vision(self) -> bool:
        return True
    
    def get_models(self) -> List[str]:
        return [
            "claude-opus-4-5",
            "claude-sonnet-4-0", 
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
        ]
    
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
        """Non-streaming completion using Anthropic API."""
        
        model_id = self.normalize_model_id(model)
        
        # Build payload
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": self.normalize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system:
            # Check if we should add cache_control
            if config.ENABLE_ANTHROPIC_CACHE_CONTROL and len(system) > 1000:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system
        
        if tools:
            anthropic_tools = self._convert_tools(tools)
            # Add cache_control to tools if enabled
            if config.ENABLE_ANTHROPIC_CACHE_CONTROL and anthropic_tools:
                for tool in anthropic_tools:
                    tool["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = anthropic_tools
        
        timeout = getattr(config, "UPSTREAM_TIMEOUT_SECONDS", 120)
        
        try:
            # Run sync SDK in thread pool
            response = await asyncio.wait_for(
                asyncio.to_thread(lambda: self.client.messages.create(**payload)),
                timeout=timeout,
            )
            
            # Extract content
            content = ""
            tool_calls = None
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })
            
            # Extract usage
            usage = response.usage
            input_tokens = usage.input_tokens if usage else 0
            output_tokens = usage.output_tokens if usage else 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            
            return CompletionResponse(
                content=content,
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=response.stop_reason or "stop",
                tool_calls=tool_calls,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )
            
        except asyncio.TimeoutError:
            log.error("Anthropic API timeout after %ds", timeout)
            raise
        except Exception as e:
            log.error("Anthropic API error: %r", e)
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
        """Streaming completion using Anthropic API."""
        
        model_id = self.normalize_model_id(model)
        
        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": self.normalize_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system:
            if config.ENABLE_ANTHROPIC_CACHE_CONTROL and len(system) > 1000:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                payload["system"] = system
        
        if tools:
            anthropic_tools = self._convert_tools(tools)
            if config.ENABLE_ANTHROPIC_CACHE_CONTROL and anthropic_tools:
                for tool in anthropic_tools:
                    tool["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = anthropic_tools
        
        try:
            # Use streaming context manager
            with self.client.messages.stream(**payload) as stream:
                current_tool_use = None
                
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            delta = event.delta
                            if hasattr(delta, "text"):
                                yield StreamChunk(content=delta.text)
                            elif hasattr(delta, "partial_json"):
                                # Tool input streaming
                                if current_tool_use:
                                    current_tool_use["arguments"] += delta.partial_json
                        
                        elif event.type == "content_block_start":
                            block = event.content_block
                            if hasattr(block, "type") and block.type == "tool_use":
                                current_tool_use = {
                                    "id": block.id,
                                    "name": block.name,
                                    "arguments": "",
                                }
                        
                        elif event.type == "content_block_stop":
                            if current_tool_use:
                                yield StreamChunk(
                                    tool_calls=[{
                                        "id": current_tool_use["id"],
                                        "type": "function",
                                        "function": {
                                            "name": current_tool_use["name"],
                                            "arguments": current_tool_use["arguments"],
                                        },
                                    }]
                                )
                                current_tool_use = None
                        
                        elif event.type == "message_stop":
                            yield StreamChunk(
                                finish_reason="stop",
                                is_final=True,
                            )
            
        except Exception as e:
            log.error("Anthropic streaming error: %r", e)
            raise
    
    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Anthropic format."""
        anthropic_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        
        return anthropic_tools
    
    def normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Normalize messages to Anthropic format."""
        normalized = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Skip system messages (handled separately)
            if role == "system":
                continue
            
            # Handle tool results
            if role == "tool":
                # Anthropic uses user role with tool_result content block
                normalized.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content if isinstance(content, str) else str(content),
                        }
                    ],
                })
                continue
            
            # Handle assistant with tool calls
            if role == "assistant" and msg.get("tool_calls"):
                content_blocks = []
                
                # Add text content if present
                if content:
                    content_blocks.append({"type": "text", "text": content})
                
                # Add tool use blocks
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": json.loads(func.get("arguments", "{}")),
                    })
                
                normalized.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
                continue
            
            # Handle content blocks (already in Anthropic format or OpenAI format)
            if isinstance(content, list):
                anthropic_blocks = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") in ["text", "image", "tool_use", "tool_result"]:
                            # Already Anthropic format
                            anthropic_blocks.append(block)
                        elif block.get("type") == "image_url":
                            # Convert OpenAI image format
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                # Parse data URL
                                import re
                                match = re.match(r"data:([^;]+);base64,(.+)", url)
                                if match:
                                    anthropic_blocks.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": match.group(1),
                                            "data": match.group(2),
                                        },
                                    })
                    elif isinstance(block, str):
                        anthropic_blocks.append({"type": "text", "text": block})
                
                normalized.append({
                    "role": role,
                    "content": anthropic_blocks if anthropic_blocks else content,
                })
            else:
                normalized.append({
                    "role": role,
                    "content": content,
                })
        
        return normalized

    def to_canonical(self, messages: List[Dict[str, Any]]) -> List[CanonicalMessage]:
        """Convert request messages to canonical format."""
        return to_canonical_messages(messages)

    def from_canonical(self, messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
        """Convert canonical messages into Anthropic-compatible format."""
        return canonical_to_anthropic_messages(messages)
