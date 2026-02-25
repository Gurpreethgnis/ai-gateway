#!/usr/bin/env python3
"""Stage 4 tests for local-context continuity helpers."""

import asyncio
import unittest
from unittest.mock import patch

from gateway import config
from gateway.providers.ollama_provider import OllamaProvider
from gateway.session_context import _deserialize_canonical, _serialize_canonical, merge_session_context
from gateway.canonical_format import to_canonical_messages


class Stage4LocalContextTests(unittest.TestCase):
    def test_session_canonical_serialization_roundtrip(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        canonical = to_canonical_messages(messages)
        encoded = _serialize_canonical(canonical)
        decoded = _deserialize_canonical(encoded)

        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0].role, "user")
        self.assertEqual(decoded[0].content[0].text, "hello")

    def test_merge_session_context_appends_and_keeps_system(self):
        stored = {
            "messages": [{"role": "user", "content": "previous"}],
            "system_prompt": "stored-system",
        }

        async def run_test():
            with patch("gateway.session_context.load_session_context", autospec=True) as mocked:
                async def _fake(_sid):
                    return stored
                mocked.side_effect = _fake

                merged_messages, merged_system = await merge_session_context(
                    session_id="abc",
                    incoming_messages=[{"role": "user", "content": "current"}],
                    incoming_system_prompt="",
                )
                return merged_messages, merged_system

        merged_messages, merged_system = asyncio.run(run_test())
        self.assertEqual(merged_system, "stored-system")
        self.assertEqual(len(merged_messages), 2)
        self.assertEqual(merged_messages[0]["content"], "previous")
        self.assertEqual(merged_messages[1]["content"], "current")

    def test_ollama_allowlist_enforced(self):
        original_url = getattr(config, "OLLAMA_URL", None)
        original_base = getattr(config, "LOCAL_LLM_BASE_URL", None)
        original_allowlist = list(getattr(config, "LOCAL_LLM_MODEL_ALLOWLIST", []))
        try:
            config.OLLAMA_URL = "http://localhost:11434"
            config.LOCAL_LLM_BASE_URL = "http://localhost:11434"
            config.LOCAL_LLM_MODEL_ALLOWLIST = ["allowed-model"]

            provider = OllamaProvider()
            provider._validate_model_allowed("allowed-model")  # should not raise
            with self.assertRaises(ValueError):
                provider._validate_model_allowed("forbidden-model")

            self.assertEqual(provider._resolve_num_ctx("missing-model"), 32768)
        finally:
            config.OLLAMA_URL = original_url
            config.LOCAL_LLM_BASE_URL = original_base
            config.LOCAL_LLM_MODEL_ALLOWLIST = original_allowlist


if __name__ == "__main__":
    unittest.main()
