# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict
import yaml
import time
import re
import logging

from .base import BaseEval

logger = logging.getLogger(__name__)


def get_retry_time(err_info):
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 1


class ModelEval(BaseEval):
    """Compute evaluate metrics using an LLM judge (OpenAI or Anthropic)."""

    def __init__(self, config: str | dict, judge_fn: Callable, format_fn: Callable):
        super().__init__()
        self.config = self.load_config(config)
        self.judge_fn = judge_fn
        self.format_fn = format_fn
        self._provider = self.config.get("provider", "openai").lower()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self._provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.get("api_key"))
            else:
                import openai
                kwargs = {"api_key": self.config.get("api_key")}
                if self.config.get("api_base"):
                    kwargs["base_url"] = self.config["api_base"]
                self._client = openai.OpenAI(**kwargs)
        return self._client

    def load_config(self, config: str | dict) -> Dict:
        if isinstance(config, dict):
            return config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    def _to_chat_messages(self, messages):
        """Ensure messages is a list of dicts (convert legacy string format if needed)."""
        if isinstance(messages, list):
            return messages
        # Legacy completion string format — wrap as a user message
        return [{"role": "user", "content": messages}]

    def chat_completion(self, messages, temperature=None, max_tokens=2000,
                        frequency_penalty=0, presence_penalty=0):
        messages = self._to_chat_messages(messages)
        if self._provider == "anthropic":
            return self._anthropic_chat(messages, temperature=temperature, max_tokens=max_tokens)
        return self._openai_chat(messages, temperature=temperature, max_tokens=max_tokens,
                                 frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

    def _openai_chat(self, messages, temperature=None, max_tokens=2000,
                     frequency_penalty=0, presence_penalty=0):
        import openai
        kwargs = {
            "model": self.config.get("model"),
            "messages": messages,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        success = False
        response = None
        while not success:
            try:
                response = self.client.chat.completions.create(**kwargs)
                success = True
            except openai.RateLimitError as e:
                logger.debug(e, exc_info=True)
                time.sleep(get_retry_time(str(e)))
            except openai.APITimeoutError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except openai.APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except openai.APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except openai.BadRequestError as e:
                logger.warning(e, exc_info=True)
                success = True
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True

        if response is None:
            return []
        try:
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            logger.warning(e, exc_info=True)
            return []

    def _anthropic_chat(self, messages, temperature=None, max_tokens=2000):
        import anthropic
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)

        kwargs = {
            "model": self.config.get("model"),
            "max_tokens": max_tokens,
            "messages": filtered,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        success = False
        response = None
        while not success:
            try:
                response = self.client.messages.create(**kwargs)
                success = True
            except anthropic.RateLimitError as e:
                logger.debug(e, exc_info=True)
                time.sleep(get_retry_time(str(e)))
            except anthropic.APIConnectionError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except anthropic.APIError as e:
                logger.debug(e, exc_info=True)
                time.sleep(1)
            except Exception as e:
                logger.warning(e, exc_info=True)
                success = True

        if response is None:
            return []
        try:
            return [block.text for block in response.content if hasattr(block, "text")]
        except Exception as e:
            logger.warning(e, exc_info=True)
            return []

    def _compute_score(self, prediction: str = None, **kwargs):
        messages = self.format_fn(prediction, chat=self.config.get("chat", True))
        response = self.chat_completion(messages, temperature=0, max_tokens=32)
        if len(response) > 0:
            return self.judge_fn(response[0])
        return -1

    def add_batch(self, *, predictions=None, **kwargs):
        batch_asrs = [self._compute_score(prediction=pred) for pred in predictions]
        self.asrs.extend(batch_asrs)
        return batch_asrs
