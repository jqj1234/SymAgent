"""Tests for LLM Client module.

Covers:
  - Initialization and config
  - generate() with mock API
  - plan_generate() with mock API
  - execute_generate() with mock API
  - extract_triples() with mock API
  - generate_with_usage() with mock API
  - API key resolution from environment
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.llm_client import LLMClient


class TestLLMClientInit(unittest.TestCase):
    """Test LLMClient initialization."""

    def test_default_init(self):
        """Test default initialization."""
        client = LLMClient()
        self.assertEqual(client.model_name, "glm-4.7-flash")
        self.assertEqual(client.temperature, 0.1)
        self.assertEqual(client.top_p, 0.9)
        self.assertEqual(client.max_new_tokens, 512)

    def test_custom_init(self):
        """Test custom initialization."""
        client = LLMClient(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            api_base="http://localhost:8080/v1",
            temperature=0.5,
            max_new_tokens=1024,
        )
        self.assertEqual(client.model_name, "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(client.temperature, 0.5)
        self.assertEqual(client.max_new_tokens, 1024)

    def test_from_config(self):
        """Test creation from config dict."""
        config = {
            "llm": {
                "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
                "api_base": "http://localhost:8000/v1",
                "temperature": 0.2,
                "top_p": 0.95,
                "max_new_tokens": 256,
            }
        }
        client = LLMClient.from_config(config)
        self.assertEqual(client.model_name, "mistralai/Mistral-7B-Instruct-v0.2")
        self.assertEqual(client.temperature, 0.2)

    def test_from_config_defaults(self):
        """Test from_config with missing keys uses defaults."""
        config = {"llm": {}}
        client = LLMClient.from_config(config)
        self.assertEqual(client.model_name, "glm-4.7-flash")
        self.assertEqual(client.temperature, 0.1)

    def test_api_key_from_env_zhipu(self):
        """Test API key resolution from ZHIPU_API_KEY env var."""
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "test_key"}):
            client = LLMClient(api_key="")
            self.assertEqual(client.client.api_key, "test_key")

    def test_api_key_from_env_openai(self):
        """Test API key resolution from OPENAI_API_KEY env var (fallback)."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ZHIPU_API_KEY if present, set OPENAI_API_KEY
            env = {"OPENAI_API_KEY": "openai_key"}
            with patch.dict(os.environ, env, clear=True):
                client = LLMClient(api_key="")
                self.assertEqual(client.client.api_key, "openai_key")

    def test_explicit_api_key_overrides_env(self):
        """Test explicit API key takes priority over env vars."""
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "env_key"}):
            client = LLMClient(api_key="explicit_key")
            self.assertEqual(client.client.api_key, "explicit_key")


class TestLLMClientGenerate(unittest.TestCase):
    """Test LLM generation methods with mocked API."""

    def _make_mock_response(self, content="test response", prompt_tokens=10, completion_tokens=5):
        """Create a mock API response."""
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = prompt_tokens
        mock_usage.completion_tokens = completion_tokens
        mock_usage.total_tokens = prompt_tokens + completion_tokens
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        return mock_response

    @patch("src.llm_client.OpenAI")
    def test_generate_basic(self, mock_openai_cls):
        """Test basic generate() call (paper Section 4 - LLM interaction)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response("Hello!")

        client = LLMClient(api_key="test")
        result = client.generate([
            {"role": "user", "content": "Hi"}
        ])

        self.assertEqual(result, "Hello!")
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "glm-4.7-flash")
        self.assertEqual(call_kwargs["temperature"], 0.1)

    @patch("src.llm_client.OpenAI")
    def test_generate_with_temperature_override(self, mock_openai_cls):
        """Test generate() with custom temperature."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response("response")

        client = LLMClient(api_key="test")
        result = client.generate([{"role": "user", "content": "test"}], temperature=0.5)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.5)

    @patch("src.llm_client.OpenAI")
    def test_generate_with_max_tokens_override(self, mock_openai_cls):
        """Test generate() with custom max_tokens."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response("response")

        client = LLMClient(api_key="test")
        result = client.generate([{"role": "user", "content": "test"}], max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["max_tokens"], 100)

    @patch("src.llm_client.OpenAI")
    def test_generate_with_stop_sequences(self, mock_openai_cls):
        """Test generate() with stop sequences."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response("partial")

        client = LLMClient(api_key="test")
        result = client.generate(
            [{"role": "user", "content": "test"}],
            stop=["Observation:"],
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["stop"], ["Observation:"])

    @patch("src.llm_client.OpenAI")
    def test_generate_strips_response(self, mock_openai_cls):
        """Test that generate() strips whitespace from response."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response("  hello world  ")

        client = LLMClient(api_key="test")
        result = client.generate([{"role": "user", "content": "test"}])
        self.assertEqual(result, "hello world")

    @patch("src.llm_client.OpenAI")
    def test_generate_propagates_error(self, mock_openai_cls):
        """Test that API errors are propagated."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        client = LLMClient(api_key="test")
        with self.assertRaises(Exception) as ctx:
            client.generate([{"role": "user", "content": "test"}])
        self.assertIn("API Error", str(ctx.exception))


class TestLLMClientPlanGenerate(unittest.TestCase):
    """Test plan_generate() method (Section 4.1 - Rule Induction)."""

    @patch("src.llm_client.OpenAI")
    def test_plan_generate_uses_system_and_user_prompts(self, mock_openai_cls):
        """Test plan_generate sends system + user messages."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[r1, r2]"))]
        )

        client = LLMClient(api_key="test")
        result = client.plan_generate(
            system_prompt="You are a planner.",
            user_prompt="Generate rules for: who directed X?",
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a planner.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(call_kwargs["temperature"], 0.3)

    @patch("src.llm_client.OpenAI")
    def test_plan_generate_default_temperature(self, mock_openai_cls):
        """Test plan_generate uses temperature=0.3 by default (Eq. 3)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="rules"))]
        )

        client = LLMClient(api_key="test")
        client.plan_generate("sys", "user")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.3)


class TestLLMClientExecuteGenerate(unittest.TestCase):
    """Test execute_generate() method (Section 4.2 - Agent Execution)."""

    @patch("src.llm_client.OpenAI")
    def test_execute_generate_single_user_message(self, mock_openai_cls):
        """Test execute_generate sends single user message."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Thought: ...\nAction: ..."))]
        )

        client = LLMClient(api_key="test")
        result = client.execute_generate("prompt text")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "prompt text")

    @patch("src.llm_client.OpenAI")
    def test_execute_generate_with_stop(self, mock_openai_cls):
        """Test execute_generate passes stop sequences."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="thought"))]
        )

        client = LLMClient(api_key="test")
        client.execute_generate("prompt", stop=["Observation:"])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["stop"], ["Observation:"])


class TestLLMClientExtractTriples(unittest.TestCase):
    """Test extract_triples() method (Section 4.2.1 - extractTriples action)."""

    @patch("src.llm_client.OpenAI")
    def test_extract_triples_sends_correct_prompt(self, mock_openai_cls):
        """Test extract_triples constructs the right prompt (Figure 6)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[[e1, r1, o1]]"))]
        )

        client = LLMClient(api_key="test")
        result = client.extract_triples(
            entity="Viggo Mortensen",
            relation="film.film.starring",
            document="Viggo Mortensen is known for Lord of the Rings.",
            question="who did viggo mortensen play in lord of the rings?",
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        content = messages[0]["content"]
        self.assertIn("Viggo Mortensen", content)
        self.assertIn("film.film.starring", content)
        self.assertIn("Lord of the Rings", content)
        self.assertIn("[[entity, relation, object]", content)
        self.assertEqual(result, "[[e1, r1, o1]]")


class TestLLMClientGenerateWithUsage(unittest.TestCase):
    """Test generate_with_usage() method."""

    @patch("src.llm_client.OpenAI")
    def test_generate_with_usage_returns_tuple(self, mock_openai_cls):
        """Test generate_with_usage returns (text, usage_dict)."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response"))],
            usage=MagicMock(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )

        client = LLMClient(api_key="test")
        text, usage = client.generate_with_usage([{"role": "user", "content": "test"}])

        self.assertEqual(text, "response")
        self.assertEqual(usage["prompt_tokens"], 20)
        self.assertEqual(usage["completion_tokens"], 10)
        self.assertEqual(usage["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
