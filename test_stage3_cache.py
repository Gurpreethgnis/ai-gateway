#!/usr/bin/env python3
"""Stage 3 cache normalization tests."""

import unittest

from gateway.prompt_cache_strategy import classify_model_tier, stabilize_system_prompt
from gateway.prompt_cache_strategy import infer_provider_from_model
from gateway.response_cache import compute_response_cache_key
from gateway.semantic_cache import compute_context_hash, extract_last_user_message


class Stage3CacheTests(unittest.TestCase):
    def test_response_cache_key_normalizes_cross_format_messages(self):
        openai_messages = [{"role": "user", "content": "hello world"}]
        anthropic_messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "hello world"}],
        }]

        key_a = compute_response_cache_key(
            messages=openai_messages,
            model="claude-sonnet-4-0",
            system="s",
            temperature=0.2,
            model_tier="smart",
        )
        key_b = compute_response_cache_key(
            messages=anthropic_messages,
            model="claude-sonnet-4-0",
            system="s",
            temperature=0.2,
            model_tier="smart",
        )

        self.assertEqual(key_a, key_b)

    def test_response_cache_key_changes_with_temperature(self):
        messages = [{"role": "user", "content": "same prompt"}]

        key_low = compute_response_cache_key(
            messages=messages,
            model="gpt-4o",
            system="s",
            temperature=0.1,
            model_tier=classify_model_tier("gpt-4o"),
        )
        key_high = compute_response_cache_key(
            messages=messages,
            model="gpt-4o",
            system="s",
            temperature=0.9,
            model_tier=classify_model_tier("gpt-4o"),
        )

        self.assertNotEqual(key_low, key_high)

    def test_semantic_helpers_use_text_from_blocks(self):
        messages = [
            {"role": "assistant", "content": "tool started"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "please summarize this"},
                    {"type": "text", "text": "and keep examples"},
                ],
            },
        ]

        query = extract_last_user_message(messages)
        self.assertIn("please summarize this", query)

        h1 = compute_context_hash(messages, "sys")
        h2 = compute_context_hash(messages, "sys")
        self.assertEqual(h1, h2)

    def test_stabilize_system_prompt_masks_dynamic_fields(self):
        src = "Run at 2026-02-25T10:22:33Z uuid 123e4567-e89b-12d3-a456-426614174000 epoch 1730000000"
        out = stabilize_system_prompt(src)
        self.assertIn("<TIMESTAMP>", out)
        self.assertIn("<UUID>", out)
        self.assertIn("<EPOCH>", out)

    def test_infer_provider_handles_plain_local_model_ids(self):
        self.assertEqual(infer_provider_from_model("llama3.2"), "ollama")
        self.assertEqual(infer_provider_from_model("qwen2.5-coder"), "ollama")
        self.assertEqual(infer_provider_from_model("local:llama3.2"), "ollama")


if __name__ == "__main__":
    unittest.main()
