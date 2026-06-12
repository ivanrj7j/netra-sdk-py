import unittest

from netra.instrumentation.litellm.utils import _set_usage_attributes

from .fixtures import OpenAI_LiteLLM_Test_Base


class TestLiteLLMProviderUtils(OpenAI_LiteLLM_Test_Base, unittest.TestCase):
    _set_usage_attributes_method = staticmethod(_set_usage_attributes)
