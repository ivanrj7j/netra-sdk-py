from typing import Any
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from .base_provider_utils import BaseProviderUtils


class OpenAI_LiteLLM_Test_Base(BaseProviderUtils):
    """Since openai and litellm have the same implementation but we need to run separate tests for these twos, we are refactoring it into a base class that implements all the methods"""

    set_request_attributes_method = None
    set_response_attributes_method = None
    _set_chat_input_method = None
    _set_response_message_attributes_method = None
    _set_usage_attributes_method = None
    _set_chat_response_input_method = None

    def _set_usage_attributes_check(self, usage_dict: dict[str, Any]) -> None:
        """
        Thoroughly verifies _set_usage_attributes mapping using internal SpanAttributes.
        Dynamically calculates and enforces exact call counts to prevent leaked writes.
        """
        mock_span = MagicMock()
        self._set_usage_attributes_method(mock_span, usage_dict)

        expected_call_count = 0

        prompt_tokens = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens")
        completion_tokens = usage_dict.get("completion_tokens") or usage_dict.get("output_tokens")

        # 1. Validate Prompt/Input Tokens
        if prompt_tokens:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
            expected_call_count += 1

        # 2. Validate Completion/Output Tokens
        if completion_tokens:
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
            expected_call_count += 1

        # 3. Validate Cached Input Pipeline
        prompt_details = usage_dict.get("prompt_tokens_details") or usage_dict.get("input_tokens_details")
        if prompt_details and isinstance(prompt_details, dict):
            if cache_tokens := prompt_details.get("cached_tokens"):
                mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, cache_tokens)
                expected_call_count += 1

        # 4. Validate Reasoning Pipeline
        completion_details = usage_dict.get("completion_tokens_details") or usage_dict.get("output_tokens_details")
        if completion_details and isinstance(completion_details, dict):
            if reasoning_tokens := completion_details.get("reasoning_tokens"):
                mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_REASONING_TOKENS, reasoning_tokens)
                expected_call_count += 1

        # 5. Validate Total Token Count
        if total_tokens := usage_dict.get("total_tokens"):
            mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
            expected_call_count += 1

        assert (
            mock_span.set_attribute.call_count == expected_call_count
        ), f"Call count mismatch! Expected exactly {expected_call_count} writes, but found {mock_span.set_attribute.call_count}."

    def test_set_usage_attributes_matrix(self) -> None:
        """
        Matrix test runner iterating over usage variations, testing distinct fields,
        aliasing formats, inner details nestings, and robust edge cases.
        """
        usage_matrix_cases = [
            # CATEGORY 1: STANDARD PRIMARY NAMING SCHEMAS
            {
                "name": "Standard_Primary_Usage_Keys",
                "payload": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
            # CATEGORY 2: ALIAS NAMING SCHEMAS
            {
                "name": "Alias_Nomenclature_Usage_Keys",
                "payload": {"input_tokens": 120, "output_tokens": 60, "total_tokens": 180},
            },
            # CATEGORY 3: INNER DETAIL OBJECT NESTINGS (Reasoning & Caching)
            {
                "name": "Primary_Details_With_Cache_And_Reasoning",
                "payload": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 35},
                },
            },
            {
                "name": "Alias_Details_With_Cache_And_Reasoning",
                "payload": {
                    "input_tokens": 250,
                    "output_tokens": 125,
                    "total_tokens": 375,
                    "input_tokens_details": {"cached_tokens": 80},
                    "output_tokens_details": {"reasoning_tokens": 55},
                },
            },
            # CATEGORY 4: ROBUSTNESS & ADVANCED EDGE CASES (Task C)
            {"name": "Edge_Case_Completely_Blank_Payload", "payload": {}},
            {
                "name": "Edge_Case_Zero_Token_Values",
                "payload": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
            {
                "name": "Edge_Case_Empty_Or_None_Details_Objects",
                "payload": {
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "prompt_tokens_details": None,
                    "completion_tokens_details": {},
                },
            },
            {
                "name": "Edge_Case_Mixed_Falsy_Tokens_With_Valid_Total",
                "payload": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 45},
            },
        ]

        # Execute all cases cleanly with isolated subtests
        for case in usage_matrix_cases:
            with self.subTest(scenario=case["name"]):
                self._set_usage_attributes_check(case["payload"])
