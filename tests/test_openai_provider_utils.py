import unittest

from netra.instrumentation.openai.utils import _set_usage_attributes

from .fixtures import OpenAI_LiteLLM_Test_Base


class TestOpenAIProviderUtils(OpenAI_LiteLLM_Test_Base, unittest.TestCase):
    _set_usage_attributes_method = staticmethod(_set_usage_attributes)
