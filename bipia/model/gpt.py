# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Any, Callable, Tuple

import re
import time
import logging

from .base import BaseModel

__all__ = ["GPTModel", "GPT35", "GPT4"]

logger = logging.getLogger(__name__)


def get_retry_time(err_info):
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 1


class GPTModel(BaseModel):
    def __init__(self, *, config: str | dict = None, **kwargs):
        config = self.load_config(config)
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            kwargs = {"api_key": self.config.get("api_key")}
            if self.config.get("api_base"):
                kwargs["base_url"] = self.config["api_base"]
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def chat_completion(self, messages, temperature=None, max_tokens=2000,
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

    def generate(self, data: Any, **kwargs):
        temperature = kwargs.pop("temperature", 0)
        rslts = []
        for message in data["message"]:
            rslt = self.chat_completion(message, temperature=temperature)
            rslts.extend(rslt)
        return rslts


class GPTModelWSystem(GPTModel):
    require_system_prompt = True

    def process_fn(self, example: Any,
                   prompt_construct_fn: Callable[[Any], Tuple[str]]) -> Any:
        system_prompt, user_prompt = prompt_construct_fn(example)
        example["message"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return example


class GPT35(GPTModelWSystem):
    pass


class GPT4(GPTModelWSystem):
    pass


class GPTModelWOSystem(GPTModel):
    require_system_prompt = False

    def process_fn(self, example: Any,
                   prompt_construct_fn: Callable[[Any], Tuple[str]]) -> Any:
        user_prompt = prompt_construct_fn(example)
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
        example["message"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return example


class GPT35WOSystem(GPTModelWOSystem):
    pass


class GPT4WOSystem(GPTModelWOSystem):
    pass
