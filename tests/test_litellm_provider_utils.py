import unittest

from netra.instrumentation.litellm.utils import (
    _set_chat_completion_input,
    _set_chat_response_input,
    _set_response_message_attributes,
    _set_usage_attributes,
    set_request_attributes,
    set_response_attributes,
)

from .fixtures import OpenAI_LiteLLM_Test_Base


class TestOpenAIProviderUtils(OpenAI_LiteLLM_Test_Base, unittest.TestCase):
    set_request_attributes_method = staticmethod(set_request_attributes)
    set_response_attributes_method = staticmethod(set_response_attributes)
    _set_chat_input_method = staticmethod(lambda span, messages, prompt: _set_chat_completion_input(span, messages))
    _set_response_message_attributes_method = staticmethod(_set_response_message_attributes)
    _set_usage_attributes_method = staticmethod(_set_usage_attributes)
    _set_chat_response_input_method = staticmethod(_set_chat_response_input)
