"""Local Model Client for SymAgent.

Wraps a HuggingFace causal LM (with optional LoRA adapter) to provide
the same generate() interface as LLMClient, enabling the self-learning
loop to switch from API-based exploration to local model exploration
after fine-tuning (Section 4.3.2).
"""

import logging
import re
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalModelClient:
    """Local HuggingFace model client compatible with LLMClient interface.

    Used after LoRA fine-tuning to replace the API-based LLM for
    online exploration in subsequent self-learning iterations.

    The agent π_θ interacts with the KG environment using this local
    model after being fine-tuned on merged trajectories D*.
    """

    def __init__(
        self,
        model_name: str,
        lora_path: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 600,
        max_new_tokens: int = 512,
        device_map: str = "auto",
    ):
        """Initialize local model client.

        Args:
            model_name: HuggingFace model name or local path.
            lora_path: Optional path to LoRA adapter directory.
                       If provided, loads the base model + LoRA adapter.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-K sampling threshold.
            max_new_tokens: Maximum tokens to generate.
            device_map: Device placement strategy.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

        logger.info(
            f"Loading local model: {model_name}"
            + (f" + LoRA: {lora_path}" if lora_path else "")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        if lora_path:
            base_model = PeftModel.from_pretrained(base_model, lora_path)
            logger.info(f"Loaded LoRA adapter from {lora_path}")

        self.model = base_model.eval()
        self.device = next(self.model.parameters()).device

        # Context window from the model config (fall back to 4096) so prompts
        # aren't needlessly truncated on long-context backbones.
        self.max_context = int(
            getattr(self.model.config, "max_position_embeddings", 4096) or 4096
        )

        logger.info(f"Local model ready on {self.device}")

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
        raw_prompt: Optional[str] = None,
    ) -> str:
        """Generate response from chat messages.

        Compatible with LLMClient.generate() interface.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Override default temperature.
            top_p: Override default top_p.
            top_k: Override default top_k.
            max_new_tokens: Override default max_new_tokens.
            stop: Optional stop sequences.
            raw_prompt: If provided, feed this string verbatim (no chat
                template). Used so executor inference matches the raw-text
                completion format the model was fine-tuned on (see
                prepare_training_data); avoids a train/inference format
                mismatch that degrades a LoRA-tuned policy.

        Returns:
            Generated text string.
        """
        temp = temperature if temperature is not None else self.temperature
        gen_top_p = top_p if top_p is not None else self.top_p
        gen_top_k = top_k if top_k is not None else self.top_k
        gen_max = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        # Raw completion (training-aligned) vs chat-template path.
        prompt = raw_prompt if raw_prompt is not None else self._format_messages(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(self.max_context - gen_max, 1),
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_max,
                temperature=temp if temp > 0 else 1.0,
                top_p=gen_top_p,
                top_k=gen_top_k,
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Apply stop sequences
        if stop:
            for s in stop:
                idx = response.find(s)
                if idx != -1:
                    response = response[:idx]

        return response.strip()

    def execute_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Single-prompt generate, compatible with LLMClient.execute_generate().

        The executor passes a fully-built ReAct prompt (Question / Thought /
        Action / Observation). We feed it verbatim as a raw completion so the
        format matches what the model was fine-tuned on
        (prepare_training_data), rather than re-wrapping it in a chat template.

        Args:
            prompt: Text prompt.
            temperature: Override temperature.
            stop: Stop sequences.

        Returns:
            Generated text.
        """
        return self.generate(
            messages=[],
            temperature=temperature,
            stop=stop,
            raw_prompt=prompt,
        )

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages into a single prompt string.

        Uses chat template if available, otherwise falls back to simple format.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: simple format
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)
