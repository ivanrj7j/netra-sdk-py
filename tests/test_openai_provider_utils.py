import json
import unittest
from typing import Any
from unittest.mock import MagicMock

from opentelemetry.semconv_ai import SpanAttributes

from netra.instrumentation.openai.utils import (
    _set_chat_completion_input,
    _set_chat_response_input,
    _set_response_message_attributes,
    _set_usage_attributes,
    set_request_attributes,
    set_response_attributes,
)

from .fixtures.base_provider_utils import BaseProviderUtils


class TestOpenAIProviderUtils(unittest.TestCase, BaseProviderUtils):
    set_request_attributes_method = staticmethod(set_request_attributes)
    set_response_attributes_method = staticmethod(set_response_attributes)
    _set_chat_input_method = staticmethod(lambda span, messages, prompt: _set_chat_completion_input(span, messages))
    _set_response_message_attributes_method = staticmethod(_set_response_message_attributes)
    _set_usage_attributes_method = staticmethod(_set_usage_attributes)

    ATTRIBUTE_MAPPINGS = BaseProviderUtils.ATTRIBUTE_MAPPINGS.copy()
    ATTRIBUTE_MAPPINGS["dimensions"] = "gen_ai.request.dimensions"

    def __set_chat_response_input_check(self, kwargs: dict[str, Any]):
        mock_span = MagicMock()

        _set_chat_response_input(mock_span, kwargs)

        message_index = 0

        if instructions := kwargs.get("instructions"):
            mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "system")
            mock_span.set_attribute.assert_any_call(
                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", instructions
            )
            message_index += 1

    def test_set_request_attributes_delegation(self):
        """Tests set_request_attributes routes correctly to completion vs response inputs"""
        mock_span = MagicMock()

        # Test 'chat' branch
        kwargs_chat = {"messages": [{"role": "user", "content": "hello"}]}
        set_request_attributes(mock_span, kwargs_chat, "chat")
        mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")

        # Test 'response' branch
        kwargs_resp = {"input": "structured input"}
        set_request_attributes(mock_span, kwargs_resp, "response")
        mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.role", "user")
        mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.0.content", "structured input")

    def test_set_request_attributes_special_fields(self):
        """Tests OpenAI-specific fields like dimensions and reasoning"""
        mock_span = MagicMock()
        kwargs = {"dimensions": 1536, "reasoning": {"depth": "high"}}
        set_request_attributes(mock_span, kwargs, "chat")

        mock_span.set_attribute.assert_any_call("gen_ai.request.dimensions", 1536)
        mock_span.set_attribute.assert_any_call(
            SpanAttributes.LLM_REQUEST_REASONING_EFFORT, json.dumps({"depth": "high"})
        )

        if input_data := kwargs.get("input"):
            if isinstance(input_data, str):
                mock_span.set_attribute.assert_any_call(f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "user")
                mock_span.set_attribute.assert_any_call(
                    f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", input_data
                )
            elif isinstance(input_data, list) and input_data:
                for message in input_data:
                    if isinstance(message, dict):
                        msg_type = message.get("type", "")
                        if msg_type == "function_call":
                            name = message.get("name", "")
                            arguments = message.get("arguments", "")
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "assistant"
                            )
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content",
                                json.dumps({"name": name, "arguments": arguments}),
                            )
                        elif msg_type == "function_call_output":
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", "tool"
                            )
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", str(message.get("output", ""))
                            )
                        else:
                            role = message.get("role", "user")
                            content = str(message.get("content", ""))
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.role", role
                            )
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_PROMPTS}.{message_index}.content", content
                            )

    def test_chat_response_input_check(self):
        # Complete matrix of test cases for verifying _set_chat_response_input
        openai_chat_response_input_cases: list[dict[str, Any]] = [
            # ---------------------------------------------------------
            # CATEGORY 1: STANDARD SUCCESS PATHS
            # ---------------------------------------------------------
            {
                "name": "Case_1_Instructions_Only",
                "description": "Tests instructions alone mapping to a single 'system' message at index 0",
                "kwargs": {"instructions": "You are a helpful and precise telemetry verification assistant."},
            },
            {
                "name": "Case_2_Raw_String_Input",
                "description": "Tests a standalone raw string input mapping to a single 'user' message at index 0",
                "kwargs": {"input": "What are the OpenTelemetry semantic conventions for LLMs?"},
            },
            {
                "name": "Case_3_Combined_Instructions_And_String_Input",
                "description": "Tests instructions followed by a raw string input, checking correct message indexing (0 and 1)",
                "kwargs": {"instructions": "Respond short and concise.", "input": "Explain Python closures."},
            },
            {
                "name": "Case_4_Standard_List_Chat_Sequence",
                "description": "Tests a structured list of conversation items under the standard fallback 'else' loop",
                "kwargs": {
                    "instructions": "Maintain an objective tone.",
                    "input": [
                        {"role": "user", "content": "Hello, can you audit this block?"},
                        {"role": "assistant", "content": "Sure, please provide the code."},
                    ],
                },
            },
            # ---------------------------------------------------------
            # CATEGORY 2: SPECIALIZED TYPE FORMATS (FUNCTION & TOOL ACTIONS)
            # ---------------------------------------------------------
            {
                "name": "Case_5_Function_Call_Interception",
                "description": "Tests type=='function_call' mapping to 'assistant' role and serialization of name/arguments",
                "kwargs": {
                    "input": [
                        {
                            "type": "function_call",
                            "name": "get_token_count",
                            "arguments": '{"provider": "openai", "model": "gpt-4o"}',
                        }
                    ]
                },
            },
            {
                "name": "Case_6_Function_Call_Output_Execution",
                "description": "Tests type=='function_call_output' mapping to 'tool' role and stringifying output",
                "kwargs": {
                    "input": [{"type": "function_call_output", "output": '{"status": "recorded", "tokens": 150}'}]
                },
            },
            {
                "name": "Case_7_Mixed_Conversational_And_Tool_Execution_Pipeline",
                "description": "Tests sequential execution indexing across system instructions, user prompts, function intercept, and tool output",
                "kwargs": {
                    "instructions": "System initialization guidelines.",
                    "input": [
                        {"role": "user", "content": "Execute database check."},
                        {
                            "type": "function_call",
                            "name": "query_db",
                            "arguments": '{"query": "SELECT count(*) FROM spans"}',
                        },
                        {"type": "function_call_output", "output": '{"count": 4200}'},
                        {"role": "assistant", "content": "The database check is complete, found 4200 spans."},
                    ],
                },
            },
            # ---------------------------------------------------------
            # CATEGORY 3: ROBUSTNESS & EXPLICIT EDGE CASES
            # ---------------------------------------------------------
            {
                "name": "Case_8_Completely_Empty_Kwargs",
                "description": "Ensures the function safely handles completely empty parameter packages without throwing KeyErrors",
                "kwargs": {},
            },
            {
                "name": "Case_9_Empty_Input_List",
                "description": "Ensures providing a blank array list skips evaluating loops cleanly",
                "kwargs": {"input": []},
            },
            {
                "name": "Case_10_Malformed_Specialized_Types_Missing_Keys",
                "description": "Verifies fallback handling inside function paths when optional fields (name, arguments, output) are absent",
                "kwargs": {
                    "input": [
                        {
                            "type": "function_call"
                            # 'name' and 'arguments' keys completely missing
                        },
                        {
                            "type": "function_call_output"
                            # 'output' key completely missing
                        },
                    ]
                },
            },
            {
                "name": "Case_11_Standard_Array_Item_Missing_Optional_Keys",
                "description": "Evaluates fallback default handling inside the general 'else' block when 'role' or 'content' is omitted",
                "kwargs": {
                    "input": [
                        {
                            # Completely empty dictionary inside the sequence list
                            # Should evaluate to role="user" and content=""
                        }
                    ]
                },
            },
            {
                "name": "Case_12_Non_String_Or_Non_Dict_Elements_In_List",
                "description": "Verifies loop defenses when an unexpected element type lands inside the input array",
                "kwargs": {
                    "input": [
                        "This string item is ignored because it's not a dictionary block",
                        12345,
                        None,
                        {"role": "user", "content": "Valid block after unexpected types."},
                    ]
                },
            },
        ]

        for case in openai_chat_response_input_cases:
            self.__set_chat_response_input_check(case)

    def __set_response_message_attributes_check(self, response_dict: dict[str, Any]) -> None:
        mock_span = MagicMock()
        _set_response_message_attributes(mock_span, response_dict)

        message_index = 0

        # 1. Path: output_text
        if output_text := response_dict.get("output_text"):
            mock_span.set_attribute.assert_any_call(
                f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant"
            )
            mock_span.set_attribute.assert_any_call(
                f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", output_text
            )
            message_index += 1

        # 2. Path: output list
        if output := response_dict.get("output"):
            for element in output:
                if element.get("type") == "function_call":
                    name = element.get("name", "")
                    arguments = element.get("arguments", "")
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant"
                    )
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content",
                        json.dumps({"name": name, "arguments": arguments}),
                    )
                    message_index += 1
                elif content := element.get("content"):
                    for chunk in content:
                        if text := chunk.get("text"):
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant"
                            )
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", text
                            )
                            message_index += 1

        # 3. Path: choices list
        if choices := response_dict.get("choices"):
            for choice in choices:
                if message := choice.get("message"):
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", message.get("role", "assistant")
                    )
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", message.get("content") or ""
                    )
                    message_index += 1

                    for tc in message.get("tool_calls") or []:
                        func = tc.get("function", {})
                        mock_span.set_attribute.assert_any_call(
                            f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant"
                        )
                        mock_span.set_attribute.assert_any_call(
                            f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content",
                            json.dumps({"name": func.get("name", ""), "arguments": func.get("arguments", "")}),
                        )
                        if tc_id := tc.get("id"):
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.tool_call_id", tc_id
                            )
                        message_index += 1

                elif delta := choice.get("delta"):
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", delta.get("role", "assistant")
                    )
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content", delta.get("content") or ""
                    )
                    message_index += 1

                    for tc in delta.get("tool_calls") or []:
                        func = tc.get("function", {})
                        mock_span.set_attribute.assert_any_call(
                            f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.role", "assistant"
                        )
                        mock_span.set_attribute.assert_any_call(
                            f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.content",
                            json.dumps({"name": func.get("name", ""), "arguments": func.get("arguments", "")}),
                        )
                        if tc_id := tc.get("id"):
                            mock_span.set_attribute.assert_any_call(
                                f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.tool_call_id", tc_id
                            )
                        message_index += 1

                if finish_reason := choice.get("finish_reason"):
                    mock_span.set_attribute.assert_any_call(
                        f"{SpanAttributes.LLM_COMPLETIONS}.{message_index}.finish_reason", finish_reason
                    )

    def test_set_response_message_attributes_matrix(self) -> None:
        """
        Matrix test runner that loops over a structured dataset of test cases,
        dynamically isolating execution spaces via subtests.
        """
        # Master list containing descriptive test scenarios and payloads
        test_cases = [
            {"name": "Unary Output Text Path", "payload": {"output_text": "Hello, how can I assist you?"}},
            {
                "name": "Choices Unary Standard Path",
                "payload": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "Parsing complete."}, "finish_reason": "stop"}
                    ]
                },
            },
            {
                "name": "Choices Live Stream Delta Chunk",
                "payload": {
                    "choices": [
                        {"delta": {"role": "assistant", "content": "fractional chunk text"}, "finish_reason": None}
                    ]
                },
            },
            {"name": "Edge Case: Blank Payload Dictionary", "payload": {}},
            {"name": "Edge Case: Missing Inner Key Objects", "payload": {"choices": [{"message": {}}]}},
        ]

        # Loops cleanly over all scenarios, mapping each execution sequence safely
        for case in test_cases:
            with self.subTest(scenario=case["name"]):
                self.__set_response_message_attributes_check(case["payload"])

    def __set_usage_attributes_check(self, usage_dict: dict[str, Any]) -> None:
        """
        Thoroughly verifies _set_usage_attributes mapping using internal SpanAttributes.
        Dynamically calculates and enforces exact call counts to prevent leaked writes.
        """
        mock_span = MagicMock()
        _set_usage_attributes(mock_span, usage_dict)

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

        # CRITICAL GUARD: Ensure no duplicate, unauthorized, or empty calls occurred
        self.assertEqual(
            mock_span.set_attribute.call_count,
            expected_call_count,
            f"Call count mismatch! Expected exactly {expected_call_count} writes, but found {mock_span.set_attribute.call_count}.",
        )

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
                self.__set_usage_attributes_check(case["payload"])
