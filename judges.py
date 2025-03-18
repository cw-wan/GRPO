import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
from accelerate import Accelerator
from huggingface_hub import InferenceClient
from transformers.utils import is_openai_available
from import_utils import is_llm_blender_available
if is_llm_blender_available():
    import llm_blender
if is_openai_available():
    from openai import OpenAI
DEFAULT_PAIRWISE_SYSTEM_PROMPT = 'I require a leaderboard for various large language models. I\'ll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.\n\n## Instruction\n\n{{\n    "instruction": """{prompt}""",\n}}\n\n## Model Outputs\n\nHere are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.\n\n{{\n    {{\n        "model_identifier": "0",\n        "output": """{response0}"""\n    }},\n    {{\n        "model_identifier": "1",\n        "output": """{response1}"""\n    }}\n}}\n\n## Task\n\nEvaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).\n'

class BaseJudge(ABC):
    @abstractmethod
    def judge(self, prompts: list[str], completions: list[str], shuffle_order: bool=True) -> list:
        raise NotImplementedError('Judge subclasses must implement the `judge` method.')

class BasePairwiseJudge(BaseJudge):
    @abstractmethod
    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool=True) -> list[int]:
        raise NotImplementedError('Judge subclasses must implement the `judge` method.')

class BaseBinaryJudge(BaseJudge):
    @abstractmethod
    def judge(self, prompts: list[str], completions: list[str], gold_completions: Optional[list[str]]=None, shuffle_order: bool=True) -> list[int]:
        raise NotImplementedError('Judge subclasses must implement the `judge` method.')