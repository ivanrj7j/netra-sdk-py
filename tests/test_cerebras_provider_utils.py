import unittest
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.cerebras.utils import set_response_attributes


class TestCerebrasProviderUtils(unittest.TestCase):
    """Unit tests for Cerebras provider utilities, focusing on token usage extraction."""

    def test_set_response_attributes_usage_success_dict(self):
        """Test successful token extraction when prompt_tokens_details is a dictionary."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        response = {
            "id": "test_id",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        }

        set_response_attributes(mock_span, response)

        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 100)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, 50)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, 150)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, 20)

    def test_set_response_attributes_usage_success_object(self):
        """Test successful token extraction when prompt_tokens_details is an object."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        class Details:
            def __init__(self, cached):
                self.cached_tokens = cached

        response = {"usage": {"prompt_tokens": 10, "prompt_tokens_details": Details(cached=5)}}

        set_response_attributes(mock_span, response)

        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 10)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, 5)

    def test_set_response_attributes_partial_usage(self):
        """Test extraction when only some usage fields are present."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        response = {"usage": {"prompt_tokens": 10}}

        set_response_attributes(mock_span, response)

        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 10)

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, called_keys)
        self.assertNotIn(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, called_keys)
        self.assertNotIn(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, called_keys)

    def test_set_response_attributes_zero_tokens(self):
        """Test that zero token values are NOT recorded (due to 'if value' checks in code),
        EXCEPT for cached_tokens which uses 'in' or 'hasattr' check."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        response = {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        }

        set_response_attributes(mock_span, response)

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, called_keys)
        self.assertNotIn(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, called_keys)
        self.assertNotIn(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, called_keys)

        # Implementation quirk: cached_tokens IS recorded if present in dict even if 0
        self.assertIn(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, called_keys)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, 0)

    def test_set_response_attributes_no_usage(self):
        """Test handling of response with missing usage field."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        response = {"id": "test"}

        set_response_attributes(mock_span, response)

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, called_keys)

    def test_set_response_attributes_not_recording(self):
        """Test that no attributes are set if the span is not recording."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        set_response_attributes(mock_span, {"usage": {"prompt_tokens": 10}})
        self.assertEqual(mock_span.set_attribute.call_count, 0)
