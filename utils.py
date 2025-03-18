import random
import json
import warnings
import dataclasses
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from collections import deque
from typing import Any, Literal, Optional, Union
from dataclasses import dataclass, field
from importlib.metadata import version
from accelerate import Accelerator, PartialState
from accelerate.state import AcceleratorState
from huggingface_hub import ModelCard, ModelCardData
from rich.console import Console
from rich.table import Table
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizerBase,
    TrainerState,
    TrainingArguments,
    is_comet_available
)
from transformers.utils import (
    is_peft_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available
)
import importlib.resources as pkg_resources
import datasets
from model_config import ModelConfig

if is_comet_available():
    import comet_ml

if is_peft_available():
    from peft import LoraConfig, PeftConfig


def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = 'right') -> torch.Tensor:
    shapes = [x.shape for x in tensors]
    biggest = np.max(shapes, axis=0).tolist()
    stacked = torch.full((len(tensors), *biggest), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)
    for idx, arr in enumerate(tensors):
        if padding_side == 'left':
            seg = slice(biggest[0] - arr.shape[0], biggest[0])
        elif padding_side == 'right':
            seg = slice(0, arr.shape[0])
        else:
            raise ValueError("padding_side必须是'left'或者'right'")
        stacked[idx][(seg,) + tuple(slice(0, d) for d in arr.shape[1:])] = arr
    return stacked


@torch.no_grad()
def get_global_statistics(accelerator, xs: torch.Tensor, mask=None, device='cpu') -> tuple[torch.Tensor, torch.Tensor, int]:
    grouped = xs.to(accelerator.device)
    sc = torch.tensor([grouped.sum(), grouped.numel() if mask is None else mask.sum()], device=grouped.device)
    sc = accelerator.reduce(sc)
    global_sum, length = sc
    g_mean = global_sum / length
    diff = (grouped - g_mean) ** 2
    if mask is not None:
        diff = diff.mul(mask)
    accum = torch.sum(diff)
    accum = accelerator.reduce(accum)
    g_var = accum / length
    return g_mean.to(device), g_var.to(device), length.item()


def get_exp_cap(value, decimal=4):
    ref = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    max_log = torch.log(ref).to(value.device)
    r = 10 ** decimal
    return torch.floor(max_log * r) / r if decimal > 0 else max_log


SIMPLE_SFT_CHAT_TEMPLATE = "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"
SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\\n\\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    ln = bools.size(-1)
    offsets = ln * (~bools).type(dtype) + torch.arange(ln, dtype=dtype, device=bools.device)
    return torch.min(offsets, dim=-1).values


def forward(model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int) -> torch.nn.Module:
    mask_t = query_responses.ne(pad_token_id)
    pos = mask_t.cumsum(dim=1) - mask_t.long()
    inps = torch.where(mask_t, query_responses, torch.zeros_like(query_responses))
    return model(
        input_ids=inps,
        attention_mask=mask_t,
        position_ids=pos,
        return_dict=True,
        output_hidden_states=True
    )


def prepare_deepspeed(model: torch.nn.Module, per_device_train_batch_size: int, fp16: bool = False, bf16: bool = False):
    import deepspeed
    ds_plugin = AcceleratorState().deepspeed_plugin
    cfg = ds_plugin.deepspeed_config
    if cfg['zero_optimization']['stage'] != 3:
        cfg['train_micro_batch_size_per_gpu'] = per_device_train_batch_size
        cfg = {
            'train_micro_batch_size_per_gpu': cfg['train_micro_batch_size_per_gpu'],
            'prescale_gradients': False,
            'wall_clock_breakdown': False
        }
        if bf16:
            cfg['bf16'] = {'enabled': True}
        elif fp16:
            cfg['fp16'] = {'enabled': True}
    elif hasattr(model, 'config'):
        hs = None
        if getattr(model.config, 'hidden_sizes', None):
            hs = max(model.config.hidden_sizes)
        else:
            hs = getattr(model.config, 'hidden_size', None)
        if hs is not None and cfg['zero_optimization']['stage'] == 3:
            cfg.update({
                'zero_optimization.reduce_bucket_size': hs * hs,
                'zero_optimization.stage3_param_persistence_threshold': 10 * hs,
                'zero_optimization.stage3_prefetch_bucket_size': 0
            })
    model, *_ = deepspeed.initialize(model=model, config=cfg)
    model.eval()
    return model


def generate(lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    ln = queries.shape[1]
    mask_c = queries.ne(pad_token_id)
    base_input = torch.where(mask_c, queries, torch.zeros_like(queries))
    out = lm_backbone.generate(
        input_ids=base_input,
        attention_mask=mask_c,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )
    logits = torch.stack(out.scores, dim=1)
    return torch.cat((queries, out.sequences[:, ln:]), dim=1), logits


def empty_cache() -> None:
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def generate_model_card(
    base_model: Optional[str],
    model_name: str,
    hub_model_id: str,
    dataset_name: Optional[str],
    tags: list[str],
    wandb_url: Optional[str],
    trainer_name: str,
    trainer_citation: Optional[str] = None,
    paper_title: Optional[str] = None,
    paper_id: Optional[str] = None,
    comet_url: Optional[str] = None
) -> ModelCard:
    card_data = ModelCardData(
        base_model=base_model,
        datasets=dataset_name,
        library_name='transformers',
        licence='license',
        model_name=model_name,
        tags=['generated_from_trainer', *tags]
    )
    path_t = str(pkg_resources.files('trl').joinpath('templates/lm_model_card.md'))
    card = ModelCard.from_template(
        card_data,
        template_path=path_t,
        base_model=base_model,
        model_name=model_name,
        hub_model_id=hub_model_id,
        dataset_name=dataset_name,
        wandb_url=wandb_url,
        comet_url=comet_url,
        trainer_name=trainer_name,
        trainer_citation=trainer_citation,
        paper_title=paper_title,
        paper_id=paper_id,
        trl_version=version('trl'),
        transformers_version=version('transformers'),
        pytorch_version=version('torch'),
        datasets_version=version('datasets'),
        tokenizers_version=version('tokenizers')
    )
    return card


def get_comet_experiment_url() -> Optional[str]:
    if not is_comet_available():
        return None
    if comet_ml.get_running_experiment() is not None:
        return comet_ml.get_running_experiment().url
    return None


def log_table_to_comet_experiment(name: str, table: pd.DataFrame) -> None:
    if not is_comet_available():
        raise ModuleNotFoundError("comet-ml 未安装。请先安装: pip install comet-ml")
    ex = comet_ml.get_running_experiment()
    if ex is not None:
        ex.log_table(tabular_data=table, filename=name)


def selective_log_softmax(logits, index):
    if logits.dtype in (torch.float32, torch.float64):
        chosen = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        sums = torch.stack([torch.logsumexp(z, dim=-1) for z in logits])
        final = chosen - sums
    else:
        coll = []
        for row_logits, row_labels in zip(logits, index):
            row_sm = F.log_softmax(row_logits, dim=-1)
            piece = row_sm.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            coll.append(piece)
        final = torch.stack(coll)
    return final
