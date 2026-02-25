#!/usr/bin/env python3
"""Unit tests for canonical message conversion helpers."""

import unittest

from gateway.canonical_format import (
    CanonicalBlock,
    CanonicalMessage,
    canonical_to_anthropic_messages,
    canonical_to_openai_messages,
    canonical_to_text_messages,
    to_canonical_messages,
)


class CanonicalFormatTests(unittest.TestCase):
    def test_openai_roundtrip_core_shapes(self):
        source = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "I will call a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"id\": 1}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result payload"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "image follows"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAA"},
                    },
                ],
            },
        ]

        canonical = to_canonical_messages(source)
        self.assertGreaterEqual(len(canonical), 5)

        back = canonical_to_openai_messages(canonical)
        roles = [m["role"] for m in back]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)

        assistant_messages = [m for m in back if m["role"] == "assistant"]
        self.assertTrue(any("tool_calls" in m for m in assistant_messages))

    def test_anthropic_conversion_has_typed_blocks(self):
        canonical = [
            CanonicalMessage(
                role="assistant",
                content=[
                    CanonicalBlock(type="text", text="hello"),
                    CanonicalBlock(type="tool_use", tool_call_id="c1", tool_name="search", arguments="{\"q\":\"x\"}"),
                ],
            ),
            CanonicalMessage(
                role="user",
                content=[
                    CanonicalBlock(type="tool_result", tool_call_id="c1", text="done"),
                ],
            ),
        ]

        anthropic_msgs = canonical_to_anthropic_messages(canonical)
        self.assertEqual(anthropic_msgs[0]["role"], "assistant")
        self.assertEqual(anthropic_msgs[0]["content"][1]["type"], "tool_use")
        self.assertEqual(anthropic_msgs[1]["content"][0]["type"], "tool_result")

    def test_text_conversion_for_local_and_groq(self):
        canonical = [
            CanonicalMessage(
                role="user",
                content=[
                    CanonicalBlock(type="text", text="question"),
                    CanonicalBlock(type="tool_result", tool_call_id="t1", text="answer"),
                ],
            )
        ]

        text_msgs = canonical_to_text_messages(canonical, include_tool_results=True, include_tool_use=False)
        self.assertEqual(len(text_msgs), 1)
        self.assertIn("[Tool result]: answer", text_msgs[0]["content"])


if __name__ == "__main__":
    unittest.main()
