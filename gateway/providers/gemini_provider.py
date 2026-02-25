"""
Gemini Provider - Google Generative AI integration.
"""

import json
from typing import List, Dict, Any, AsyncIterator, Optional

from gateway.canonical_format import CanonicalMessage, canonical_to_openai_messages, to_canonical_messages
from gateway.providers.base import BaseProvider, CompletionResponse, StreamChunk
from gateway.logging_setup import log
from gateway import config


class GeminiProvider(BaseProvider):
    """Google Gemini API provider."""
    
    def __init__(self):
        self.api_key = getattr(config, "GEMINI_API_KEY", None)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        self._client = None
    
    @property
    def name(self) -> str:
        return "gemini"
    
    @property
    def client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._client
    
    def supports_tools(self) -> bool:
        return True
    
    def supports_vision(self) -> bool:
        return True
    
    def get_models(self) -> List[str]:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
    
    def _normalize_system_instruction(self, system: Any) -> Optional[str]:
        """Normalize system prompt to a plain string. Accepts Anthropic-style blocks (type, text, cache_control)."""
        if system is None:
            return None
        if isinstance(system, str):
            return system.strip() or None
        if isinstance(system, list):
            parts = []
            for block in system:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", "") or "")
                    elif "text" in block:
                        parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "\n\n".join(p for p in parts if p).strip() or None
        return str(system).strip() or None

    def _convert_messages_to_gemini(
        self,
        messages: List[Dict],
        system: Optional[str] = None,
    ) -> tuple:
        """Convert OpenAI-style messages to Gemini format."""
        contents = []
        system_instruction = self._normalize_system_instruction(system)
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Extract system from messages if not provided
            if role == "system":
                if not system_instruction:
                    system_instruction = content if isinstance(content, str) else str(content)
                continue
            
            # Map roles
            gemini_role = "user" if role == "user" else "model"
            
            # Handle content
            if isinstance(content, str):
                parts = [content]
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "image_url":
                            image_url = (block.get("image_url") or {}).get("url") or ""
                            if image_url.startswith("data:") and "," in image_url:
                                header, data = image_url.split(",", 1)
                                mime = "image/png"
                                if ";" in header:
                                    mime = header.replace("data:", "").split(";")[0] or "image/png"
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime,
                                        "data": data,
                                    }
                                })
                        elif block.get("type") == "image":
                            # Handle image - Gemini uses inline_data
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                parts.append({
                                    "inline_data": {
                                        "mime_type": source.get("media_type", "image/png"),
                                        "data": source.get("data", ""),
                                    }
                                })
                    elif isinstance(block, str):
                        parts.append(block)
            else:
                parts = [str(content)]
            
            contents.append({
                "role": gemini_role,
                "parts": parts,
            })
        
        return contents, system_instruction
    
    # JSON Schema keys that Gemini's Schema type does not support (causes ValueError if present)
    _GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset({
        "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
        "pattern", "default", "examples", "exclusiveMinimum", "exclusiveMaximum",
    })

    def _sanitize_schema_for_gemini(self, obj: Any) -> Any:
        """Recursively remove JSON Schema keys that Gemini API does not accept."""
        if isinstance(obj, dict):
            return {
                k: self._sanitize_schema_for_gemini(v)
                for k, v in obj.items()
                if k not in self._GEMINI_UNSUPPORTED_SCHEMA_KEYS
            }
        if isinstance(obj, list):
            return [self._sanitize_schema_for_gemini(item) for item in obj]
        return obj

    def _convert_tools_to_gemini(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Gemini function declarations."""
        if not tools:
            return []

        functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                parameters = func.get("parameters") or {}
                parameters = self._sanitize_schema_for_gemini(parameters)
                functions.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": parameters,
                })

        return [{"function_declarations": functions}] if functions else []
    
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
        """Non-streaming completion using Gemini API."""
        import asyncio
        
        model_id = self.normalize_model_id(model)
        contents, system_instruction = self._convert_messages_to_gemini(messages, system)
        
        # Configure generation
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Create model instance
        model_kwargs = {"generation_config": generation_config}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        
        gemini_model = self.client.GenerativeModel(model_id, **model_kwargs)
        
        # Add tools if provided
        gemini_tools = self._convert_tools_to_gemini(tools) if tools else None
        
        try:
            # Run in thread pool since Gemini SDK is sync
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                contents,
                tools=gemini_tools,
            )
            
            content = ""
            tool_calls = None
            
            if response.candidates:
                candidate = response.candidates[0]
                parts = candidate.content.parts if candidate.content else []
                
                for part in parts:
                    if hasattr(part, "text"):
                        content += part.text
                    elif hasattr(part, "function_call"):
                        # Handle function calls
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args)),
                            },
                        })
            
            # Get usage if available
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
            
            return CompletionResponse(
                content=content,
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason="stop",
                tool_calls=tool_calls,
            )
            
        except Exception as e:
            log.error("Gemini API error: %r", e)
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
        """Streaming completion using Gemini API."""
        import asyncio
        
        model_id = self.normalize_model_id(model)
        contents, system_instruction = self._convert_messages_to_gemini(messages, system)
        
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        
        model_kwargs = {"generation_config": generation_config}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        
        gemini_model = self.client.GenerativeModel(model_id, **model_kwargs)
        gemini_tools = self._convert_tools_to_gemini(tools) if tools else None
        
        try:
            # Generate with streaming
            response = await asyncio.to_thread(
                lambda: gemini_model.generate_content(
                    contents,
                    tools=gemini_tools,
                    stream=True,
                )
            )
            
            for chunk in response:
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text"):
                                yield StreamChunk(
                                    content=part.text,
                                    is_final=False,
                                )
            
            yield StreamChunk(
                content="",
                finish_reason="stop",
                is_final=True,
            )
            
        except Exception as e:
            log.error("Gemini streaming error: %r", e)
            raise
    
    def normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Pass through - conversion happens in _convert_messages_to_gemini."""
        return messages

    def to_canonical(self, messages: List[Dict[str, Any]]) -> List[CanonicalMessage]:
        """Convert request messages to canonical format."""
        return to_canonical_messages(messages)

    def from_canonical(self, messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
        """Convert canonical messages into OpenAI-compatible format for Gemini ingestion."""
        return canonical_to_openai_messages(messages)
