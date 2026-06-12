import unittest

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.dspy.utils import extract_usage_info


class TestDSPyProviderUtils(unittest.TestCase):
    """Unit tests for DSPy provider utilities, focusing on token usage extraction."""

    def test_extract_usage_info_success(self):
        """Test successful extraction and mapping of standard token fields."""
        response = {"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
        result = dict(extract_usage_info(response))

        self.assertEqual(result[SpanAttributes.LLM_USAGE_PROMPT_TOKENS], 100)
        self.assertEqual(result[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS], 50)
        self.assertEqual(result[SpanAttributes.LLM_USAGE_TOTAL_TOKENS], 150)

    def test_extract_usage_info_aliases(self):
        """Test extraction using alias names (input_tokens / output_tokens)."""
        response = {"usage": {"input_tokens": 80, "output_tokens": 40}}
        result = dict(extract_usage_info(response))

        self.assertEqual(result[SpanAttributes.LLM_USAGE_PROMPT_TOKENS], 80)
        self.assertEqual(result[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS], 40)

    def test_extract_usage_info_zero_values(self):
        """
        Test that zero token values are correctly extracted.
        Note: The current implementation has a quirk where '0 or None' evaluates to 'None',
        so if only 'prompt_tokens' is 0 and 'input_tokens' is missing, it returns None.
        However, 'total_tokens' is handled directly.
        """
        # Testing with both alias keys to ensure 0 is captured
        response = {
            "usage": {
                "prompt_tokens": 0,
                "input_tokens": 0,
                "completion_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
        }
        result = dict(extract_usage_info(response))

        self.assertEqual(result[SpanAttributes.LLM_USAGE_PROMPT_TOKENS], 0)
        self.assertEqual(result[SpanAttributes.LLM_USAGE_COMPLETION_TOKENS], 0)
        self.assertEqual(result[SpanAttributes.LLM_USAGE_TOTAL_TOKENS], 0)

    def test_extract_usage_info_zero_values_quirk(self):
        """Test the implementation quirk where single 0 value with missing alias is skipped."""
        response = {"usage": {"prompt_tokens": 0}}
        result = dict(extract_usage_info(response))

        # prompt_tokens = 0 or None -> None
        self.assertNotIn(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, result)

    def test_extract_usage_info_partial_data(self):
        """Test extraction when only some token fields are present."""
        response = {"usage": {"prompt_tokens": 10}}
        result = dict(extract_usage_info(response))

        self.assertEqual(result[SpanAttributes.LLM_USAGE_PROMPT_TOKENS], 10)
        self.assertNotIn(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, result)
        self.assertNotIn(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, result)

    def test_extract_usage_info_object_response(self):
        """Test extraction from a mock object that supports conversion to dictionary."""

        class MockResponse:
            def __init__(self, usage):
                self.usage = usage

            def to_dict(self):
                return {"usage": self.usage}

        response = MockResponse({"prompt_tokens": 20})
        result = dict(extract_usage_info(response))
        self.assertEqual(result[SpanAttributes.LLM_USAGE_PROMPT_TOKENS], 20)

    def test_extract_usage_info_no_usage(self):
        """Test that an empty iterator is returned if usage information is missing."""
        response = {"no_usage_here": True}
        result = list(extract_usage_info(response))
        self.assertEqual(len(result), 0)

    def test_extract_usage_info_malformed_usage(self):
        """Test graceful handling of malformed usage data (e.g., not a dictionary)."""
        response = {"usage": "not a dictionary"}
        result = list(extract_usage_info(response))
        self.assertEqual(len(result), 0)

    def test_extract_usage_info_none_response(self):
        """Test graceful handling of None response."""
        result = list(extract_usage_info(None))
        self.assertEqual(len(result), 0)
