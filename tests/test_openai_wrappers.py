"""Tests for OpenAI instrumentation streaming wrappers."""

import asyncio
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.openai.wrappers import AsyncStreamingWrapper, StreamingWrapper


class TestOpenAIWrappers(unittest.TestCase):
    """Test suite for validating synchronous and asynchronous OpenAI streaming wrappers."""

    def __streaming_wrapper_check(
        self,
        fake_chunks: list[dict[str, Any]],
        fake_request_kwargs: dict[str, Any],
        expected_netra_output: str,
        expected_prompt_tokens: int,
        expected_completion_tokens: int,
    ):
        """
        Verify that the synchronous StreamingWrapper correctly captures output and usage data.

        Args:
            fake_chunks: List of mock response chunks.
            fake_request_kwargs: Mock request keyword arguments.
            expected_netra_output: The expected stitched output string.
            expected_prompt_tokens: The expected prompt token count in span attributes.
            expected_completion_tokens: The expected completion token count in span attributes.
        """
        mock_span = MagicMock()

        # Using a MagicMock as the stream to bypass an SDK bug where attributes
        # are incorrectly set on the wrapped object instead of the proxy itself.
        # MagicMock allows attribute assignment, whereas list_iterator (iter([])) does not.
        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = mock_stream
        mock_stream.__next__.side_effect = fake_chunks

        wrapper = StreamingWrapper(mock_span, mock_stream, fake_request_kwargs)

        for _ in wrapper:
            pass

        self.assertEqual(wrapper._netra_output, expected_netra_output)

        if expected_prompt_tokens > 0:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, expected_prompt_tokens)

        if expected_completion_tokens > 0:
            mock_span.set_attribute.assert_any_call(
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, expected_completion_tokens
            )

    async def __async_streaming_wrapper_check(
        self,
        fake_chunks: list[dict[str, Any]],
        fake_request_kwargs: dict[str, Any],
        expected_netra_output: str,
        expected_prompt_tokens: int,
        expected_completion_tokens: int,
    ):
        """
        Verify that the AsyncStreamingWrapper correctly captures output and usage data.

        Args:
            fake_chunks: List of mock response chunks.
            fake_request_kwargs: Mock request keyword arguments.
            expected_netra_output: The expected stitched output string.
            expected_prompt_tokens: The expected prompt token count in span attributes.
            expected_completion_tokens: The expected completion token count in span attributes.
        """
        mock_span = MagicMock()

        mock_stream = MagicMock()
        mock_stream.__aiter__.return_value = mock_stream
        # AsyncMock.side_effect handles the list of chunks and then raises StopAsyncIteration
        mock_stream.__anext__ = AsyncMock(side_effect=fake_chunks + [StopAsyncIteration])

        wrapper = AsyncStreamingWrapper(mock_span, mock_stream, fake_request_kwargs)

        async for _ in wrapper:
            pass

        self.assertEqual(wrapper._netra_output, expected_netra_output)

        if expected_prompt_tokens > 0:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, expected_prompt_tokens)

        if expected_completion_tokens > 0:
            mock_span.set_attribute.assert_any_call(
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, expected_completion_tokens
            )

    def test_streaming_scenarios(self):
        """Run multiple streaming scenarios using a data-driven approach."""
        scenarios = self._get_scenarios()

        for case in scenarios:
            # SDK replaces usage data with each new chunk (last one wins)
            expected_prompt = 0
            expected_completion = 0
            for chunk in case["chunks"]:
                if usage := chunk.get("usage"):
                    expected_prompt = usage.get("prompt_tokens", 0)
                    expected_completion = usage.get("completion_tokens", 0)

            with self.subTest(case=case["name"]):
                self.__streaming_wrapper_check(
                    fake_chunks=case["chunks"],
                    fake_request_kwargs=case["kwargs"],
                    expected_netra_output=case["expected_output"],
                    expected_prompt_tokens=expected_prompt,
                    expected_completion_tokens=expected_completion,
                )

    def test_async_streaming_scenarios(self):
        """Run multiple async streaming scenarios."""
        scenarios = self._get_scenarios()

        for case in scenarios:
            expected_prompt = 0
            expected_completion = 0
            for chunk in case["chunks"]:
                if usage := chunk.get("usage"):
                    expected_prompt = usage.get("prompt_tokens", 0)
                    expected_completion = usage.get("completion_tokens", 0)

            with self.subTest(case=case["name"]):
                asyncio.run(
                    self.__async_streaming_wrapper_check(
                        fake_chunks=case["chunks"],
                        fake_request_kwargs=case["kwargs"],
                        expected_netra_output=case["expected_output"],
                        expected_prompt_tokens=expected_prompt,
                        expected_completion_tokens=expected_completion,
                    )
                )

    def _get_scenarios(self):
        """
        Return a list of test scenarios for streaming wrappers.

        Each scenario is a dictionary containing:
        - name: Description of the test case.
        - kwargs: Arguments passed to the mock request.
        - chunks: A list of mock response chunks to be streamed.
        - expected_output: The expected final stitched output string.
        """
        return [
            {
                "name": "Standard Success Path",
                "kwargs": {"stream": True, "messages": []},
                "chunks": [
                    {"choices": [{"index": 0, "delta": {"content": "Hello "}}]},
                    {
                        "choices": [{"index": 0, "delta": {"content": "World"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    },
                ],
                "expected_output": "Hello World",
            },
            {
                "name": "Empty Usage (Edge Case)",
                "kwargs": {"stream": True, "messages": []},
                "chunks": [{"choices": [{"index": 0, "delta": {"content": "No usage here"}}]}],
                "expected_output": "No usage here",
            },
            {
                "name": "Accumulating Multi-Chunk Usage",
                "kwargs": {"stream": True, "messages": []},
                "chunks": [
                    {"choices": [{"index": 0, "delta": {"content": "A"}}], "usage": {"prompt_tokens": 5}},
                    {
                        "choices": [{"index": 0, "delta": {"content": "B"}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
                    },
                ],
                "expected_output": "AB",
            },
            {
                "name": "Stitching Tool Call Arguments",
                "kwargs": {"stream": True, "messages": []},
                "chunks": [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{"index": 0, "function": {"name": "func", "arguments": '{"loc'}}]
                                },
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ation": "NYC"}'}}]},
                            }
                        ]
                    },
                ],
                "expected_output": "",
            },
            {
                "name": "Multi-Choice Handling",
                "kwargs": {"stream": True, "messages": []},
                "chunks": [
                    {
                        "choices": [
                            {"index": 0, "delta": {"content": "Opt1"}},
                            {"index": 1, "delta": {"content": "Opt2"}},
                        ]
                    },
                ],
                "expected_output": "Opt1Opt2",
            },
            {
                "name": "Malformed Usage Dictionary",
                "kwargs": {"stream": True, "messages": []},
                "chunks": [
                    {"usage": {}},  # Edge case: key exists but is empty
                    {"choices": [{"index": 0, "delta": {"content": "StillWorks"}}]},
                ],
                "expected_output": "StillWorks",
            },
            {
                "name": "Finish Reason Capture",
                "kwargs": {"stream": True, "messages": []},
                "chunks": [{"choices": [{"index": 0, "delta": {"content": "Done"}, "finish_reason": "stop"}]}],
                "expected_output": "Done",
            },
        ]
