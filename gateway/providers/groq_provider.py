"""
Groq Provider - Fast inference API integration.
"""

from typing import List, Dict, Any, AsyncIterator, Optional

from gateway.canonical_format import CanonicalMessage, canonical_to_text_messages, to_canonical_messages
from gateway.providers.base import BaseProvider, CompletionResponse, StreamChunk
from gateway.logging_setup import log
from gateway import config


class GroqProvider(BaseProvider):
    """Groq API provider for fast LLM inference."""
    
    def __init__(self):
        self.api_key = getattr(config, "GROQ_API_KEY", None)
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured")
        
        self._client = None
    
    @property
    def name(self) -> str:
        return "groq"
    
    @property
    def client(self):
        """Lazy-load Groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self.api_key)
            except ImportError:
                raise ImportError("groq package not installed. Run: pip install groq")
        return self._client
    
    def supports_tools(self) -> bool:
        return True
    
    def supports_vision(self) -> bool:
        return False  # Groq doesn't support vision yet
    
    def get_models(self) -> List[str]:
        return [
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral-8x7b-32768",
            "groq/gemma2-9b-it",
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
        """Non-streaming completion using Groq API."""
        
        # Normalize model ID (remove groq/ prefix if present)
        model_id = self.normalize_model_id(model)
        
        # Build messages with system prompt
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(self.normalize_messages(messages))
        
        request_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add tools if provided
        if tools:
            request_kwargs["tools"] = tools
        
        try:
            response = await self.client.chat.completions.create(**request_kwargs)
            
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Extract tool calls
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            
            return CompletionResponse(
                content=content,
                model=model_id,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                finish_reason=choice.finish_reason or "stop",
                tool_calls=tool_calls,
            )
            
        except Exception as e:
            log.error("Groq API error: %r", e)
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
        """Streaming completion using Groq API."""
        
        model_id = self.normalize_model_id(model)
        
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(self.normalize_messages(messages))
        
        request_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        if tools:
            request_kwargs["tools"] = tools
        
        try:
            stream = await self.client.chat.completions.create(**request_kwargs)
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                content = delta.content or ""
                finish_reason = choice.finish_reason
                
                # Handle tool calls
                tool_calls = None
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = [
                        {
                            "index": tc.index,
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name if tc.function else None,
                                "arguments": tc.function.arguments if tc.function else "",
                            },
                        }
                        for tc in delta.tool_calls
                    ]
                
                yield StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                    is_final=finish_reason is not None,
                )
                
        except Exception as e:
            log.error("Groq streaming error: %r", e)
            raise
    
    def normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Normalize messages to Groq/OpenAI format."""
        normalized = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle tool results
            if role == "tool":
                normalized.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else str(content),
                })
                continue
            
            # Handle assistant with tool calls
            if role == "assistant" and msg.get("tool_calls"):
                normalized.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": msg["tool_calls"],
                })
                continue
            
            # Handle content blocks - flatten to text
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            
            normalized.append({
                "role": role,
                "content": content,
            })
        
        return normalized

    def to_canonical(self, messages: List[Dict[str, Any]]) -> List[CanonicalMessage]:
        """Convert request messages to canonical format."""
        return to_canonical_messages(messages)

    def from_canonical(self, messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
        """Convert canonical messages to Groq-safe text messages."""
        return canonical_to_text_messages(messages, include_tool_results=True, include_tool_use=True)
