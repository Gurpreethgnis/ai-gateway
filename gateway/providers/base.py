"""
Base Provider Abstract Class.

All model providers (Anthropic, OpenAI, Gemini, Groq, Ollama) implement
this interface for consistent request handling.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator, Optional, AsyncGenerator
from dataclasses import dataclass


@dataclass
class CompletionResponse:
    """Standardized completion response."""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    tool_calls: Optional[List[Dict]] = None
    raw_response: Optional[Dict] = None
    
    # Provider-specific metadata
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class StreamChunk:
    """Chunk from a streaming response."""
    content: str = ""
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    is_final: bool = False
    
    # For SSE formatting
    def to_sse(self, model: str) -> str:
        """Format as Server-Sent Event."""
        import json
        data = {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": self.finish_reason,
            }]
        }
        
        if self.content:
            data["choices"][0]["delta"]["content"] = self.content
        
        if self.tool_calls:
            data["choices"][0]["delta"]["tool_calls"] = self.tool_calls
        
        return f"data: {json.dumps(data)}\n\n"


class BaseProvider(ABC):
    """
    Abstract base class for model providers.
    
    Each provider implements:
    - complete(): Non-streaming completion
    - stream(): Streaming completion
    - supports_tools(): Whether tools/functions are supported
    - get_models(): List available models
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'anthropic', 'openai')."""
        pass
    
    @abstractmethod
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
        """
        Non-streaming completion.
        
        Args:
            messages: Conversation messages
            model: Model ID
            system: System prompt (optional)
            tools: Tool definitions (optional)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            **kwargs: Provider-specific options
            
        Returns:
            CompletionResponse with content and metadata
        """
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Streaming completion.
        
        Args:
            Same as complete()
            
        Yields:
            StreamChunk objects with incremental content
        """
        pass
    
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports tool/function calling."""
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider supports vision/image inputs."""
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """List of model IDs this provider can serve."""
        pass
    
    def normalize_model_id(self, model: str) -> str:
        """
        Normalize model ID for API call.
        
        Strips provider prefix if present (e.g., 'ollama/llama3.2' -> 'llama3.2').
        """
        if "/" in model:
            return model.split("/", 1)[1]
        return model
    
    def normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Normalize messages to provider's expected format.
        
        Override in subclasses for provider-specific transformations.
        """
        return messages
