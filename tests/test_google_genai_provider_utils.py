import unittest
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.google_genai.utils import set_response_attributes


class TestGoogleGenAIProviderUtils(unittest.TestCase):
    """Unit tests for Google GenAI provider utilities, focusing on token usage extraction."""

    def test_set_usage_attributes_success(self):
        """Test successful extraction and mapping of all token attributes."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_usage = MagicMock()
        mock_usage.total_token_count = 150
        mock_usage.candidates_token_count = 40
        mock_usage.thoughts_token_count = 10
        mock_usage.prompt_token_count = 100
        mock_usage.cached_content_token_count = 20

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = None  # Skip candidate content processing

        set_response_attributes(mock_span, mock_response)

        # Verify exact mappings and summation for completion tokens
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, 150)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, 50)  # 40 + 10
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 100)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, 20)

    def test_set_usage_attributes_partial_data(self):
        """Test extraction when only some token fields are present in the metadata."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        # mock_usage using spec to ensure only specific attributes exist
        class MockUsage:
            def __init__(self, prompt):
                self.prompt_token_count = prompt

        mock_usage = MockUsage(prompt=50)
        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = None

        set_response_attributes(mock_span, mock_response)

        # Verify present field
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 50)

        # Verify absent fields were not recorded
        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, called_keys)
        self.assertNotIn(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, called_keys)
        self.assertNotIn(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, called_keys)

    def test_set_usage_attributes_zero_values(self):
        """Test that zero values are recorded (except for completion tokens which requires > 0)."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_usage = MagicMock()
        mock_usage.total_token_count = 0
        mock_usage.candidates_token_count = 0
        mock_usage.thoughts_token_count = 0
        mock_usage.prompt_token_count = 0
        mock_usage.cached_content_token_count = 0

        mock_response = MagicMock()
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = None

        set_response_attributes(mock_span, mock_response)

        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_TOTAL_TOKENS, 0)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 0)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS, 0)

        # Completion tokens should NOT be recorded if sum is not > 0
        called_keys = [call[0][0] for call in mock_span.set_attribute.call_args_list]
        self.assertNotIn(SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, called_keys)

    def test_extract_usage_from_chunk_dict(self):
        """Test extraction of usage metadata from a dictionary response containing a 'chunk'."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 75

        mock_chunk = MagicMock()
        mock_chunk.usage_metadata = mock_usage

        # Simulating a streaming chunk structure
        response = {"chunk": mock_chunk}

        set_response_attributes(mock_span, response)
        mock_span.set_attribute.assert_any_call(SpanAttributes.LLM_USAGE_PROMPT_TOKENS, 75)

    def test_set_response_attributes_not_recording(self):
        """Test that no attributes are set if the span is not recording."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        set_response_attributes(mock_span, MagicMock())
        self.assertEqual(mock_span.set_attribute.call_count, 0)

    def test_set_response_attributes_none_response(self):
        """Test graceful handling of None response."""
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        set_response_attributes(mock_span, None)
        self.assertEqual(mock_span.set_attribute.call_count, 0)
