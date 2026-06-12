import unittest
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.pydantic_ai.utils import set_pydantic_response_attributes


class TestPydanticAIProviderUtils(unittest.TestCase):
    """Unit tests for Pydantic AI provider utilities, focusing on token usage extraction."""

    def test_set_pydantic_response_attributes_usage_success(self):
        """Test successful extraction and mapping of token attributes from a Pydantic AI result."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_usage = MagicMock()
        mock_usage.request_tokens = 120
        mock_usage.response_tokens = 60
        mock_usage.total_tokens = 180
        mock_usage.details = {"cache_hit": 10}

        mock_result = MagicMock()
        mock_result.usage.return_value = mock_usage
        mock_result.model_name = "pydantic-ai-model"
        mock_result.output = None  # Focus on usage

        set_pydantic_response_attributes(mock_span, mock_result)

        # Verify standard token mappings
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, "120")
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, "60")
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, "180")

        # Verify additional details mapping
        mock_span.set_attribute.assert_any_call("gen_ai.usage.details.cache_hit", "10")

    def test_set_pydantic_response_attributes_partial_usage(self):
        """Test extraction when only some usage fields are present."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_usage = MagicMock(spec=["request_tokens", "total_tokens"])
        mock_usage.request_tokens = 50
        mock_usage.total_tokens = 50
        mock_usage.details = None

        mock_result = MagicMock()
        mock_result.usage.return_value = mock_usage
        mock_result.model_name = None
        mock_result.output = None

        set_pydantic_response_attributes(mock_span, mock_result)

        # Verify present fields (values are converted to string by _safe_set_attribute)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, "50")
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, "50")

        # Verify absent fields were not recorded
        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, called_keys)

    def test_set_pydantic_response_attributes_zero_tokens(self):
        """Test that zero token values are correctly recorded."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_usage = MagicMock()
        mock_usage.request_tokens = 0
        mock_usage.response_tokens = 0
        mock_usage.total_tokens = 0
        mock_usage.details = {}

        mock_result = MagicMock()
        mock_result.usage.return_value = mock_usage

        set_pydantic_response_attributes(mock_span, mock_result)

        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, "0")
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, "0")
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, "0")

    def test_set_pydantic_response_attributes_no_usage(self):
        """Test handling of result with no usage information."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_result = MagicMock(spec=["model_name", "output"])
        # No 'usage' attribute here

        set_pydantic_response_attributes(mock_span, mock_result)

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, called_keys)

    def test_set_pydantic_response_attributes_empty_usage(self):
        """Test handling of result where usage() returns None."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_result = MagicMock()
        mock_result.usage.return_value = None

        set_pydantic_response_attributes(mock_span, mock_result)

        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, called_keys)
