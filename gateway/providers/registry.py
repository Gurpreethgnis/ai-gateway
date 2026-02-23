"""
Provider Registry - Centralized provider management.

Auto-discovers and manages all available providers based on
configured API keys.
"""

from typing import Dict, Optional, List, TYPE_CHECKING

from gateway.logging_setup import log
from gateway.providers.base import BaseProvider

if TYPE_CHECKING:
    from gateway.model_registry import ModelInfo


class ProviderRegistry:
    """
    Registry of available model providers.
    
    Discovers providers based on configured API keys and maintains
    provider instances for request handling.
    """
    
    _instance: Optional["ProviderRegistry"] = None
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize and discover available providers."""
        if self._initialized:
            return
        
        from gateway import config
        
        # Anthropic
        if config.ANTHROPIC_API_KEY:
            try:
                from gateway.providers.anthropic_provider import AnthropicProvider
                self.providers["anthropic"] = AnthropicProvider()
                log.info("Registered provider: anthropic")
            except Exception as e:
                log.warning("Failed to initialize Anthropic provider: %r", e)
        
        # OpenAI
        openai_key = getattr(config, "OPENAI_API_KEY", None)
        if openai_key:
            try:
                from gateway.providers.openai_provider import OpenAIProvider
                self.providers["openai"] = OpenAIProvider()
                log.info("Registered provider: openai")
            except Exception as e:
                log.warning("Failed to initialize OpenAI provider: %r", e)
        
        # Gemini
        gemini_key = getattr(config, "GEMINI_API_KEY", None)
        if gemini_key:
            try:
                from gateway.providers.gemini_provider import GeminiProvider
                self.providers["gemini"] = GeminiProvider()
                log.info("Registered provider: gemini")
            except Exception as e:
                log.warning("Failed to initialize Gemini provider: %r", e)
        
        # Groq
        groq_key = getattr(config, "GROQ_API_KEY", None)
        if groq_key:
            try:
                from gateway.providers.groq_provider import GroqProvider
                self.providers["groq"] = GroqProvider()
                log.info("Registered provider: groq")
            except Exception as e:
                log.warning("Failed to initialize Groq provider: %r", e)
        
        # Ollama (local)
        ollama_url = getattr(config, "OLLAMA_URL", None) or getattr(config, "LOCAL_LLM_URL", None)
        if ollama_url:
            try:
                from gateway.providers.ollama_provider import OllamaProvider
                self.providers["ollama"] = OllamaProvider()
                log.info("Registered provider: ollama")
            except Exception as e:
                log.warning("Failed to initialize Ollama provider: %r", e)
        
        self._initialized = True
        log.info("ProviderRegistry initialized with %d providers", len(self.providers))
    
    def get(self, provider_name: str) -> Optional[BaseProvider]:
        """Get a provider by name."""
        return self.providers.get(provider_name)
    
    def get_for_model(self, model_id: str) -> Optional[BaseProvider]:
        """Get the provider for a specific model ID."""
        from gateway.model_registry import get_model_registry
        
        registry = get_model_registry()
        model = registry.get_model(model_id)
        
        if model:
            return self.providers.get(model.provider)
        
        # Fallback: check if model ID has provider prefix
        if "/" in model_id:
            provider_name = model_id.split("/")[0]
            return self.providers.get(provider_name)
        
        # Try to infer from model name patterns
        if model_id.startswith("claude"):
            return self.providers.get("anthropic")
        if model_id.startswith("gpt") or model_id.startswith("o1"):
            return self.providers.get("openai")
        if model_id.startswith("gemini"):
            return self.providers.get("gemini")
        
        return None
    
    def is_available(self, provider_name: str) -> bool:
        """Check if a provider is available."""
        return provider_name in self.providers
    
    def list_providers(self) -> List[str]:
        """List available provider names."""
        return list(self.providers.keys())
    
    def list_all_models(self) -> List[str]:
        """List all models from all providers."""
        models = []
        for provider in self.providers.values():
            models.extend(provider.get_models())
        return models


# =============================================================================
# Global Instance
# =============================================================================

_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


async def initialize_provider_registry() -> ProviderRegistry:
    """Initialize the global provider registry."""
    registry = get_provider_registry()
    await registry.initialize()
    return registry


def get_provider(provider_name: str) -> Optional[BaseProvider]:
    """Convenience function to get a provider."""
    return get_provider_registry().get(provider_name)


def get_provider_for_model(model_id: str) -> Optional[BaseProvider]:
    """Convenience function to get provider for a model."""
    return get_provider_registry().get_for_model(model_id)
