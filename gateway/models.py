from typing import Any, Optional, List, Dict, Literal, Union
from pydantic import BaseModel, Field

from gateway.config import DEFAULT_MAX_TOKENS

ProviderType = Literal["anthropic", "local"]

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatReq(BaseModel):
    system: Optional[str] = ""
    messages: List[ChatMessage]
    max_tokens: int = DEFAULT_MAX_TOKENS
    model: Optional[str] = None
    temperature: Optional[float] = 0.2
    provider: Optional[ProviderType] = None  # "local" for Ollama, default/None for Anthropic

OpenAIRole = Literal["system", "user", "assistant", "tool", "developer"]
OAContent = Union[str, List[Any], Dict[str, Any], None]

class OAChatMessage(BaseModel):
    role: OpenAIRole
    content: OAContent = None

    class Config:
        extra = "allow"

class OAChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[OAChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False

    class Config:
        extra = "allow"
