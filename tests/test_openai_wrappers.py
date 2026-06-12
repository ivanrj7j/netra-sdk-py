"""Tests for OpenAI instrumentation streaming wrappers focusing on token accumulation."""

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.openai.wrappers import AsyncStreamingWrapper, StreamingWrapper


class TestOpenAIWrappers(unittest.TestCase):
    """Test suite for validating token usage accumulation in streaming wrappers."""

    def __streaming_wrapper_check(
        self, fake_chunks: list[dict[str, Any]], expected_prompt: int, expected_completion: int
    ):
        """Verify synchronous token capture."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = mock_stream
        mock_stream.__next__.side_effect = fake_chunks

        wrapper = StreamingWrapper(mock_span, mock_stream, {"stream": True})
        for _ in wrapper:
            pass

        if expected_prompt > 0:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, expected_prompt)
        if expected_completion > 0:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, expected_completion)

    async def __async_streaming_wrapper_check(
        self, fake_chunks: list[dict[str, Any]], expected_prompt: int, expected_completion: int
    ):
        """Verify asynchronous token capture."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_stream = MagicMock()
        mock_stream.__aiter__.return_value = mock_stream
        mock_stream.__anext__ = AsyncMock(side_effect=fake_chunks + [StopAsyncIteration])

        wrapper = AsyncStreamingWrapper(mock_span, mock_stream, {"stream": True})
        async for _ in wrapper:
            pass

        if expected_prompt > 0:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, expected_prompt)
        if expected_completion > 0:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, expected_completion)

    def test_streaming_token_scenarios(self):
        """Run token-focused scenarios."""
        scenarios = [
            {
                "name": "Last Chunk Contains Usage",
                "chunks": [
                    {"choices": [{"index": 0, "delta": {"content": "Hi"}}]},
                    {"choices": [{"index": 0, "delta": {}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                ],
                "expected_prompt": 10,
                "expected_completion": 5,
            },
            {
                "name": "Mid-Stream and Final Usage (Last wins)",
                "chunks": [
                    {"usage": {"prompt_tokens": 5}},
                    {"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
                ],
                "expected_prompt": 10,
                "expected_completion": 20,
            },
            {
                "name": "Empty Usage",
                "chunks": [{"choices": [{"index": 0, "delta": {"content": "No usage here"}}]}],
                "expected_prompt": 0,
                "expected_completion": 0,
            },
        ]

        for case in scenarios:
            with self.subTest(case=case["name"]):
                # Sync test
                self.__streaming_wrapper_check(case["chunks"], case["expected_prompt"], case["expected_completion"])
                # Async test
                asyncio.run(
                    self.__async_streaming_wrapper_check(
                        case["chunks"], case["expected_prompt"], case["expected_completion"]
                    )
                )
