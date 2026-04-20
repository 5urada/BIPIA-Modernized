# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml
import logging
from pathlib import Path
from collections import OrderedDict

from .gpt import GPT35, GPT4, GPT35WOSystem, GPT4WOSystem

logger = logging.getLogger(__name__)

LLM_NAME_TO_CLASS = OrderedDict(
    [
        ("gpt35", GPT35),
        ("gpt4", GPT4),
        ("gpt35_wosys", GPT35WOSystem),
        ("gpt4_wosys", GPT4WOSystem),
    ]
)


class AutoLLM:
    @classmethod
    def from_name(cls, name: str):
        if name in LLM_NAME_TO_CLASS:
            pass
        elif Path(name).exists():
            with open(name, "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            if "llm_name" not in config:
                raise ValueError("llm_name not in config.")
            name = config["llm_name"]
        else:
            raise ValueError(
                f"Invalid name {name}. AutoLLM.from_name needs llm name or llm config as inputs."
            )

        logger.info(f"Load {name} from name.")
        return LLM_NAME_TO_CLASS[name]
