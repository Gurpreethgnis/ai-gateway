"""
Model Registry - Centralized model metadata and capabilities.

Stores information about all available models across providers including
cost, latency, quality ratings, and capabilities. Can be customized
via dashboard or database overrides.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

from gateway.logging_setup import log


@dataclass
class ModelInfo:
    """Information about a model's capabilities and characteristics."""
    
    # Identity
    id: str                         # Unique model ID (e.g., "claude-sonnet-4-0")
    provider: str                   # Provider name (e.g., "anthropic")
    display_name: str               # Human-readable name
    
    # Capabilities
    supports_tools: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    context_window: int = 128000
    max_output_tokens: int = 4096
    
    # Performance characteristics
    cost_per_1m_input: float = 0.0      # USD per 1M input tokens
    cost_per_1m_output: float = 0.0     # USD per 1M output tokens
    avg_latency_ms: int = 500           # Average response latency
    quality_rating: float = 0.75        # 0-1 calibrated quality estimate
    code_quality_boost: float = 1.0     # Multiplier for code tasks
    
    # User control
    is_enabled: bool = True
    priority: int = 100                 # Lower = higher priority (for tie-breaking)
    
    # Metadata
    last_updated: Optional[datetime] = None
    
    @property
    def normalized_cost(self) -> float:
        """Normalized cost score (0-1) for comparison."""
        max_cost = 20.0  # Approximate max expected
        if self.cost_per_1m_input <= 0:
            return 0.0
        return min(self.cost_per_1m_input / max_cost, 1.0)
    
    @property
    def normalized_latency(self) -> float:
        """Normalized latency score (0-1) for comparison."""
        max_latency = 2000  # 2 seconds
        return min(self.avg_latency_ms / max_latency, 1.0)


# =============================================================================
# Default Model Definitions
# =============================================================================

DEFAULT_MODELS: List[ModelInfo] = [
    # -------------------------------------------------------------------------
    # Anthropic
    # -------------------------------------------------------------------------
    ModelInfo(
        id="claude-opus-4-5",
        provider="anthropic",
        display_name="Claude Opus 4.5",
        supports_tools=True,
        supports_vision=True,
        context_window=200000,
        max_output_tokens=32000,
        cost_per_1m_input=15.0,
        cost_per_1m_output=75.0,
        avg_latency_ms=1200,
        quality_rating=0.95,
        code_quality_boost=1.15,
        priority=10,
    ),
    ModelInfo(
        id="claude-sonnet-4-0",
        provider="anthropic",
        display_name="Claude Sonnet 4",
        supports_tools=True,
        supports_vision=True,
        context_window=200000,
        max_output_tokens=16000,
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
        avg_latency_ms=600,
        quality_rating=0.88,
        code_quality_boost=1.15,
        priority=20,
    ),
    ModelInfo(
        id="claude-3-5-haiku-20241022",
        provider="anthropic",
        display_name="Claude 3.5 Haiku",
        supports_tools=True,
        supports_vision=True,
        context_window=200000,
        max_output_tokens=8192,
        cost_per_1m_input=0.80,
        cost_per_1m_output=4.0,
        avg_latency_ms=300,
        quality_rating=0.78,
        code_quality_boost=1.05,
        priority=30,
    ),
    
    # -------------------------------------------------------------------------
    # OpenAI
    # -------------------------------------------------------------------------
    ModelInfo(
        id="gpt-4o",
        provider="openai",
        display_name="GPT-4o",
        supports_tools=True,
        supports_vision=True,
        context_window=128000,
        max_output_tokens=16384,
        cost_per_1m_input=5.0,
        cost_per_1m_output=15.0,
        avg_latency_ms=800,
        quality_rating=0.92,
        code_quality_boost=1.0,
        priority=15,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        display_name="GPT-4o Mini",
        supports_tools=True,
        supports_vision=True,
        context_window=128000,
        max_output_tokens=16384,
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        avg_latency_ms=400,
        quality_rating=0.78,
        code_quality_boost=1.0,
        priority=35,
    ),
    ModelInfo(
        id="o1",
        provider="openai",
        display_name="OpenAI o1",
        supports_tools=False,  # o1 doesn't support tools yet
        supports_vision=True,
        context_window=200000,
        max_output_tokens=100000,
        cost_per_1m_input=15.0,
        cost_per_1m_output=60.0,
        avg_latency_ms=5000,  # Reasoning takes time
        quality_rating=0.96,
        code_quality_boost=1.20,
        priority=5,
        is_enabled=False,  # Disabled by default (expensive, slow)
    ),
    
    # -------------------------------------------------------------------------
    # Google Gemini
    # -------------------------------------------------------------------------
    ModelInfo(
        id="gemini-1.5-pro",
        provider="gemini",
        display_name="Gemini 1.5 Pro",
        supports_tools=True,
        supports_vision=True,
        context_window=2000000,  # 2M context!
        max_output_tokens=8192,
        cost_per_1m_input=3.50,
        cost_per_1m_output=10.50,
        avg_latency_ms=700,
        quality_rating=0.87,
        code_quality_boost=1.0,
        priority=25,
    ),
    ModelInfo(
        id="gemini-1.5-flash",
        provider="gemini",
        display_name="Gemini 1.5 Flash",
        supports_tools=True,
        supports_vision=True,
        context_window=1000000,
        max_output_tokens=8192,
        cost_per_1m_input=0.075,
        cost_per_1m_output=0.30,
        avg_latency_ms=300,
        quality_rating=0.75,
        code_quality_boost=1.0,
        priority=40,
    ),
    ModelInfo(
        id="gemini-2.0-flash",
        provider="gemini",
        display_name="Gemini 2.0 Flash",
        supports_tools=True,
        supports_vision=True,
        context_window=1000000,
        max_output_tokens=8192,
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40,
        avg_latency_ms=250,
        quality_rating=0.80,
        code_quality_boost=1.05,
        priority=38,
    ),
    
    # -------------------------------------------------------------------------
    # Groq (Fast inference)
    # -------------------------------------------------------------------------
    ModelInfo(
        id="groq/llama-3.3-70b-versatile",
        provider="groq",
        display_name="Llama 3.3 70B (Groq)",
        supports_tools=True,
        supports_vision=False,
        context_window=128000,
        max_output_tokens=32768,
        cost_per_1m_input=0.59,
        cost_per_1m_output=0.79,
        avg_latency_ms=150,  # Groq is fast!
        quality_rating=0.82,
        code_quality_boost=1.0,
        priority=45,
    ),
    ModelInfo(
        id="groq/llama-3.1-8b-instant",
        provider="groq",
        display_name="Llama 3.1 8B (Groq)",
        supports_tools=True,
        supports_vision=False,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1m_input=0.05,
        cost_per_1m_output=0.08,
        avg_latency_ms=80,
        quality_rating=0.70,
        code_quality_boost=1.0,
        priority=55,
    ),
    ModelInfo(
        id="groq/mixtral-8x7b-32768",
        provider="groq",
        display_name="Mixtral 8x7B (Groq)",
        supports_tools=True,
        supports_vision=False,
        context_window=32768,
        max_output_tokens=8192,
        cost_per_1m_input=0.24,
        cost_per_1m_output=0.24,
        avg_latency_ms=100,
        quality_rating=0.76,
        code_quality_boost=1.05,
        priority=50,
    ),
    
    # -------------------------------------------------------------------------
    # Ollama (Local - Free)
    # -------------------------------------------------------------------------
    ModelInfo(
        id="ollama/qwen2.5-coder:32b-instruct",
        provider="ollama",
        display_name="Qwen 2.5 Coder 32B",
        supports_tools=False,
        supports_vision=False,
        context_window=32768,
        max_output_tokens=8192,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=800,
        quality_rating=0.78,
        code_quality_boost=1.25,
        priority=60,
    ),
    ModelInfo(
        id="ollama/qwen2.5-coder:14b-instruct",
        provider="ollama",
        display_name="Qwen 2.5 Coder 14B",
        supports_tools=False,
        supports_vision=False,
        context_window=32768,
        max_output_tokens=8192,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=500,
        quality_rating=0.72,
        code_quality_boost=1.20,
        priority=65,
    ),
    ModelInfo(
        id="ollama/qwen2.5-coder:7b-instruct",
        provider="ollama",
        display_name="Qwen 2.5 Coder 7B",
        supports_tools=False,
        supports_vision=False,
        context_window=32768,
        max_output_tokens=8192,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=300,
        quality_rating=0.65,
        code_quality_boost=1.15,
        priority=70,
    ),
    ModelInfo(
        id="ollama/llama3.2:latest",
        provider="ollama",
        display_name="Llama 3.2",
        supports_tools=False,
        supports_vision=False,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=400,
        quality_rating=0.68,
        code_quality_boost=1.0,
        priority=75,
    ),
    ModelInfo(
        id="ollama/llama3.1:8b",
        provider="ollama",
        display_name="Llama 3.1 8B",
        supports_tools=False,
        supports_vision=False,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=250,
        quality_rating=0.62,
        code_quality_boost=1.0,
        priority=80,
    ),
    ModelInfo(
        id="ollama/deepseek-coder-v2:16b",
        provider="ollama",
        display_name="DeepSeek Coder V2 16B",
        supports_tools=False,
        supports_vision=False,
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=600,
        quality_rating=0.74,
        code_quality_boost=1.20,
        priority=62,
    ),
    ModelInfo(
        id="ollama/qwen2.5:14b",
        provider="ollama",
        display_name="Qwen 2.5 14B",
        supports_tools=False,
        supports_vision=False,
        context_window=32768,
        max_output_tokens=8192,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=450,
        quality_rating=0.70,
        code_quality_boost=1.0,
        priority=72,
    ),
]


# =============================================================================
# Model Registry Class
# =============================================================================

class ModelRegistry:
    """
    Registry of all available models across providers.
    
    Supports:
    - Default model definitions
    - Database overrides for enabled/disabled state
    - Provider availability filtering
    - Dynamic model discovery (Ollama)
    """
    
    _instance: Optional["ModelRegistry"] = None
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._load_defaults()
        self._available_providers: set = set()
    
    def _load_defaults(self):
        """Load default model definitions."""
        for model in DEFAULT_MODELS:
            self.models[model.id] = model
    
    async def initialize(self):
        """
        Initialize registry with runtime data.
        
        - Discovers available providers based on API keys
        - Loads model settings from database
        - Discovers local Ollama models
        """
        from gateway import config
        
        # Discover providers based on configured API keys
        if config.ANTHROPIC_API_KEY:
            self._available_providers.add("anthropic")
            log.info("Provider available: anthropic")
        
        if getattr(config, "OPENAI_API_KEY", None):
            self._available_providers.add("openai")
            log.info("Provider available: openai")
        
        if getattr(config, "GEMINI_API_KEY", None):
            self._available_providers.add("gemini")
            log.info("Provider available: gemini")
        
        if getattr(config, "GROQ_API_KEY", None):
            self._available_providers.add("groq")
            log.info("Provider available: groq")
        
        if getattr(config, "OLLAMA_URL", None) or getattr(config, "LOCAL_LLM_URL", None):
            self._available_providers.add("ollama")
            log.info("Provider available: ollama")
            await self._discover_ollama_models()
        
        # Load database overrides
        await self._load_db_settings()
        
        log.info(
            "ModelRegistry initialized: %d models, %d providers",
            len(self.models),
            len(self._available_providers),
        )
    
    async def _discover_ollama_models(self):
        """Discover locally available Ollama models."""
        from gateway import config
        import httpx
        
        ollama_url = getattr(config, "OLLAMA_URL", None) or getattr(config, "LOCAL_LLM_URL", None)
        if not ollama_url:
            return
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("models", []):
                        name = model.get("name", "")
                        model_id = f"ollama/{name}"
                        
                        # Skip if already defined
                        if model_id in self.models:
                            continue
                        
                        # Add dynamically discovered model
                        self.models[model_id] = ModelInfo(
                            id=model_id,
                            provider="ollama",
                            display_name=f"Ollama {name}",
                            supports_tools=False,
                            supports_vision=False,
                            context_window=32768,  # Conservative default
                            cost_per_1m_input=0.0,
                            cost_per_1m_output=0.0,
                            avg_latency_ms=500,
                            quality_rating=0.65,  # Conservative default
                            priority=90,
                        )
                        log.debug("Discovered Ollama model: %s", model_id)
        except Exception as e:
            log.warning("Failed to discover Ollama models: %r", e)
    
    async def _load_db_settings(self):
        """Load model settings overrides from database."""
        from gateway.config import DATABASE_URL
        
        if not DATABASE_URL:
            return
        
        try:
            from gateway.db import get_session
            from sqlalchemy import text
            
            async with get_session() as session:
                # Check if model_settings table exists
                result = await session.execute(
                    text("""
                        SELECT model_id, is_enabled, custom_quality_rating
                        FROM model_settings
                        WHERE project_id IS NULL
                    """)
                )
                
                for row in result.fetchall():
                    model_id, is_enabled, custom_quality = row
                    if model_id in self.models:
                        self.models[model_id].is_enabled = is_enabled
                        if custom_quality is not None:
                            self.models[model_id].quality_rating = custom_quality
                        log.debug(
                            "Loaded DB settings for %s: enabled=%s, quality=%s",
                            model_id, is_enabled, custom_quality
                        )
        except Exception as e:
            # Table might not exist yet
            if "model_settings" not in str(e).lower():
                log.debug("Could not load model settings from DB: %r", e)
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model by ID."""
        return self.models.get(model_id)
    
    def get_enabled_models(self) -> List[ModelInfo]:
        """Get all enabled models from available providers."""
        return [
            m for m in self.models.values()
            if m.is_enabled and m.provider in self._available_providers
        ]

    async def get_enabled_models_for_project(self, project_id: Optional[int]) -> List[ModelInfo]:
        """
        Get enabled models, applying project-specific overrides from model_settings.
        When project_id is None, uses global overrides from model_settings (project_id IS NULL) so
        dashboard toggles are respected across all workers.
        When project_id is set, uses that project's overrides so per-project toggles are respected.
        """
        base = [
            m for m in self.models.values()
            if m.provider in self._available_providers
        ]
        from gateway.config import DATABASE_URL
        if not DATABASE_URL:
            return [m for m in base if m.is_enabled]
        try:
            from gateway.db import get_session
            from sqlalchemy import text
            async with get_session() as session:
                if project_id is None:
                    result = await session.execute(
                        text("""
                            SELECT model_id, is_enabled
                            FROM model_settings
                            WHERE project_id IS NULL
                        """)
                    )
                else:
                    result = await session.execute(
                        text("""
                            SELECT model_id, is_enabled
                            FROM model_settings
                            WHERE project_id = :project_id
                        """),
                        {"project_id": project_id},
                    )
                overrides = {row[0]: bool(row[1]) for row in result.fetchall()}
        except Exception as e:
            log.debug("Could not load model settings: %r", e)
            return [m for m in base if m.is_enabled]
        # If no overrides, use in-memory is_enabled
        out = []
        for m in base:
            effective = overrides.get(m.id, m.is_enabled)
            if effective:
                out.append(m)
        return out

    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Get all models for a specific provider."""
        return [m for m in self.models.values() if m.provider == provider]
    
    def get_provider_for_model(self, model_id: str) -> Optional[str]:
        """Get the provider name for a model ID."""
        model = self.models.get(model_id)
        return model.provider if model else None
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available."""
        return provider in self._available_providers
    
    def set_model_enabled(self, model_id: str, enabled: bool) -> bool:
        """Enable or disable a model."""
        if model_id in self.models:
            self.models[model_id].is_enabled = enabled
            return True
        return False
    
    def update_model_quality(self, model_id: str, quality: float) -> bool:
        """Update a model's quality rating."""
        if model_id in self.models:
            self.models[model_id].quality_rating = max(0.0, min(1.0, quality))
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary for API responses."""
        return {
            "models": {
                m.id: {
                    "id": m.id,
                    "provider": m.provider,
                    "display_name": m.display_name,
                    "supports_tools": m.supports_tools,
                    "supports_vision": m.supports_vision,
                    "context_window": m.context_window,
                    "cost_per_1m_input": m.cost_per_1m_input,
                    "cost_per_1m_output": m.cost_per_1m_output,
                    "avg_latency_ms": m.avg_latency_ms,
                    "quality_rating": m.quality_rating,
                    "is_enabled": m.is_enabled,
                    "provider_available": m.provider in self._available_providers,
                }
                for m in self.models.values()
            },
            "providers": {
                p: {"available": p in self._available_providers}
                for p in {"anthropic", "openai", "gemini", "groq", "ollama"}
            },
        }


# =============================================================================
# Global Instance
# =============================================================================

_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


async def initialize_model_registry():
    """Initialize the global model registry."""
    registry = get_model_registry()
    await registry.initialize()
    return registry
