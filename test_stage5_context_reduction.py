#!/usr/bin/env python3
"""Stage 5 tests for intelligent context reduction behavior."""

import unittest
from unittest.mock import patch

from gateway.context_pruner import (
    score_message_importance,
    resolve_effective_context_limit,
)
from gateway.token_reduction import strip_or_truncate
from gateway.context_compression import COMPRESSION_MODEL


class Stage5ContextReductionTests(unittest.TestCase):
    def test_importance_scoring_prioritizes_critical_and_recent(self):
        total = 3
        old_msg = {"role": "user", "content": "hello"}
        critical_msg = {"role": "user", "content": "critical security bug fix required"}

        score_old = score_message_importance(old_msg, 0, total)
        score_critical = score_message_importance(critical_msg, 1, total)

        self.assertGreater(score_critical, score_old)

    def test_resolve_effective_context_limit_uses_model_registry(self):
        class FakeModel:
            context_window = 10000

        class FakeRegistry:
            def get_model(self, model_id):
                if model_id == "claude-sonnet-4-0":
                    return FakeModel()
                return None

        with patch("gateway.context_pruner.get_model_registry", create=True):
            with patch("gateway.model_registry.get_model_registry", return_value=FakeRegistry()):
                limit = resolve_effective_context_limit(
                    max_tokens=60000,
                    provider="anthropic",
                    model="claude-sonnet-4-0",
                )

        self.assertEqual(limit, 8000)  # 80% of 10000

    def test_truncation_marker_includes_omitted_count(self):
        long_text = "x" * 1000
        reduced, meta = strip_or_truncate("user", long_text, max_chars=200, allow_strip=False)

        self.assertTrue(meta.get("truncated"))
        self.assertIn("TRUNCATED:", reduced)
        self.assertIn("chars omitted", reduced)

    def test_compression_model_comes_from_config_module(self):
        self.assertTrue(COMPRESSION_MODEL)


if __name__ == "__main__":
    unittest.main()
