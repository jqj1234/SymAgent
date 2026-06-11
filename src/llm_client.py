"""
LLM Client for SymAgent.

Supports multiple LLM backends via OpenAI-compatible API:
  - Zhipu (智谱) API: glm-4.7-flash
  - vLLM-served local models
  - OpenAI and other compatible endpoints

Provides a unified interface for both planning and execution prompts.
"""

import logging
import os
import time
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# Default settings for Zhipu API
ZHIPU_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_DEFAULT_MODEL = "glm-4.7-flash"


class LLMClient:
    """Unified LLM client supporting OpenAI-compatible APIs.

    Works with Zhipu, vLLM, OpenAI, or any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model_name: str = ZHIPU_DEFAULT_MODEL,
        api_base: str = ZHIPU_DEFAULT_BASE_URL,
        api_key: str = "",
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 600,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ):
        # Resolve API key: explicit arg > env var ZHIPU_API_KEY > env var OPENAI_API_KEY
        if not api_key:
            api_key = os.environ.get("ZHIPU_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "EMPTY")

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.extra_kwargs = kwargs
        self._min_interval = kwargs.pop("min_request_interval", 2.0)
        self._last_request_time = 0.0
        self._max_retries = kwargs.pop("max_retries", 5)
        self._slow_retry_wait = kwargs.pop("slow_retry_wait", 180)

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        logger.info(
            f"LLMClient initialized: model={model_name}, "
            f"api_base={api_base}, temperature={temperature}"
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Override default temperature.
            max_tokens: Override default max_new_tokens.
            stop: Optional stop sequences.

        Returns:
            Generated text string.
        """
        retry_count = kwargs.pop("_retry_count", 0)
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "top_p": self.top_p,
        }
        if stop:
            params["stop"] = stop
        params.update(kwargs)

        try:
            self._rate_limit_wait()
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            is_slow_retry = (
                "429" in error_str
                or "tpm limit" in error_lower
                or "rate limit" in error_lower
                or "timed out" in error_lower
                or "timeout" in error_lower
            )
            if retry_count < self._max_retries:
                if is_slow_retry:
                    wait = self._slow_retry_wait
                    logger.warning(
                        "LLM request hit rate limit/timeout, waiting %ss before retry "
                        "(attempt %s/%s): %s",
                        wait,
                        retry_count + 1,
                        self._max_retries,
                        e,
                    )
                else:
                    wait = 2 ** (retry_count + 1)
                    logger.warning(
                        "LLM request failed, retrying in %ss (attempt %s/%s): %s",
                        wait,
                        retry_count + 1,
                        self._max_retries,
                        e,
                    )
                time.sleep(wait)
                return self.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    _retry_count=retry_count + 1,
                    **kwargs,
                )
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_with_usage(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int]]:
        """Generate a completion and return token usage info.

        Returns:
            Tuple of (generated_text, usage_dict).
        """
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_new_tokens,
            "top_p": self.top_p,
        }
        if stop:
            params["stop"] = stop
        params.update(kwargs)

        response = self.client.chat.completions.create(**params)
        text = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return text, usage

    def plan_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """Generate planning output (symbolic rules).

        Uses slightly higher temperature for creative rule induction.

        Args:
            system_prompt: System-level instruction for planning.
            user_prompt: User-level prompt with question and demonstrations.
            temperature: Temperature for generation.

        Returns:
            Generated symbolic rules as text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(messages, temperature=temperature)

    def execute_generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate execution output (thought-action pairs).

        Args:
            prompt: Full prompt including history and available actions.
            temperature: Temperature for generation (low for deterministic).
            stop: Stop sequences (e.g., ["Observation:"]).

        Returns:
            Generated thought and action text.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate(messages, temperature=temperature, stop=stop)

    def extract_triples(
        self,
        entity: str,
        relation: str,
        document: str,
        question: str,
        temperature: float = 0.1,
    ) -> str:
        """Extract triples from document using the extraction prompt template.

        Corresponds to the extractTriples action from the paper (Figure 6).

        Args:
            entity: The entity to extract triples about.
            relation: The target relation.
            document: The retrieved document text.
            question: The original question.

        Returns:
            Extracted triples in list format.
        """
        extraction_prompt = (
            f"Here is the document about the entity {entity}:\n"
            f"{document}. "
            f"You should extract relevant factual triples about {entity} "
            f"under the relation {relation}, which are beneficial to answer "
            f"the question {question}. "
            f"You should only output the triples in the form of "
            f"[[entity, relation, object], ...]"
        )
        messages = [{"role": "user", "content": extraction_prompt}]
        return self.generate(messages, temperature=temperature)

    def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between requests to avoid 429 rate limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LLMClient":
        """Create an LLMClient from a configuration dict.

        Supports automatic API key resolution from environment variables.

        Args:
            config: Configuration dictionary with LLM settings.

        Returns:
            Initialized LLMClient instance.
        """
        llm_config = config.get("llm", {})
        return cls(
            model_name=llm_config.get("model_name", ZHIPU_DEFAULT_MODEL),
            api_base=llm_config.get("api_base", ZHIPU_DEFAULT_BASE_URL),
            api_key=llm_config.get("api_key", ""),
            temperature=llm_config.get("temperature", 0.1),
            top_p=llm_config.get("top_p", 0.9),
            top_k=llm_config.get("top_k", 600),
            max_new_tokens=llm_config.get("max_new_tokens", 512),
        )
