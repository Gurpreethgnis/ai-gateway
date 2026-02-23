"""
OpenAI Provider - Native OpenAI API integration.
"""

from typing import List, Dict, Any, AsyncIterator, Optional

from gateway.providers.base import BaseProvider, CompletionResponse, StreamChunk
from gateway.logging_setup import log
from gateway import config


class OpenAIProvider(BaseProvider):
    """OpenAI API provider for GPT models."""
    
    def __init__(self):
        self.api_key = getattr(config, "OPENAI_API_KEY", None)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        
        self._client = None
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        return self._client
    
    def supports_tools(self) -> bool:
        return True
    
    def supports_vision(self) -> bool:
        return True
    
    def get_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "gpt-4-turbo"]
    
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
        """Non-streaming completion using OpenAI API."""
        
        model_id = self.normalize_model_id(model)
        
        # Build messages with system prompt
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(self.normalize_messages(messages))
        
        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        
        # o1 models don't support temperature or system messages the same way
        if not model_id.startswith("o1"):
            request_kwargs["temperature"] = temperature
        
        # Add tools if provided
        if tools and not model_id.startswith("o1"):
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
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )
            
        except Exception as e:
            log.error("OpenAI API error: %r", e)
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
        """Streaming completion using OpenAI API."""
        
        model_id = self.normalize_model_id(model)
        
        # Build messages with system prompt
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(self.normalize_messages(messages))
        
        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        if not model_id.startswith("o1"):
            request_kwargs["temperature"] = temperature
        
        if tools and not model_id.startswith("o1"):
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
                
                # Handle tool calls in stream
                tool_calls = None
                if delta.tool_calls:
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
            log.error("OpenAI streaming error: %r", e)
            raise
    
    def normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Normalize messages to OpenAI format."""
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
            
            # Handle content blocks (Anthropic format -> OpenAI)
            if isinstance(content, list):
                # Convert to OpenAI content format
                openai_content = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            openai_content.append({
                                "type": "text",
                                "text": block.get("text", ""),
                            })
                        elif block.get("type") == "image":
                            # Handle image content
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                openai_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}",
                                    },
                                })
                    elif isinstance(block, str):
                        openai_content.append({"type": "text", "text": block})
                
                normalized.append({
                    "role": role,
                    "content": openai_content if openai_content else content,
                })
            else:
                normalized.append({
                    "role": role,
                    "content": content,
                })
        
        return normalized
