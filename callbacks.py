import os
from typing import List, Optional, Union
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import (
    gather_object,
    is_comet_ml_available,
    is_deepspeed_available,
    is_wandb_available
)
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments
)
from transformers.trainer_utils import has_length
from data_utils import maybe_apply_chat_template
from import_utils import is_mergekit_available
from mergekit_utils import MergeConfig, merge_models, upload_model_to_hf
from models.utils import unwrap_model_for_generation
from judges import BasePairwiseJudge
from utils import log_table_to_comet_experiment
if is_deepspeed_available():
    import deepspeed
if is_comet_ml_available():
    pass
if is_wandb_available():
    import wandb

def _generate_completions(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    accelerator: Accelerator,
    generation_config: Optional[GenerationConfig],
    batch_size: int = 1
) -> list[str]:
    results_list = []
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        total_prompts = len(prompts)
        for start_idx in range(0, total_prompts, batch_size):
            prompt_slice = prompts[start_idx:start_idx + batch_size]
            tokenized_data = tokenizer(
                prompt_slice,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(model.device)
            generated_outputs = unwrapped_model.generate(
                **tokenized_data,
                generation_config=generation_config
            )
            for prompt_ids, gen_seq in zip(tokenized_data.input_ids, generated_outputs):
                gen_seq = gen_seq[len(prompt_ids):]
                decoded_text = tokenizer.decode(gen_seq, skip_special_tokens=True)
                results_list.append(decoded_text)
    return results_list

class SyncRefModelCallback(TrainerCallback):
    def __init__(
        self,
        ref_model: Union[PreTrainedModel, torch.nn.Module],
        accelerator: Optional[Accelerator]
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_target_model(base_model, target_model, alpha_value):
        # 通过对参数进行加权更新来实现同步
        for ref_param, main_param in zip(target_model.parameters(), base_model.parameters()):
            ref_param.data.mul_(1.0 - alpha_value).add_(main_param.data, alpha=alpha_value)

    @staticmethod
    def sync_target_model(base_model, target_model, alpha_value):
        # 根据 deepspeed 的配置确定是否需要收集全部参数再同步
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        if deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3:
            with deepspeed.zero.GatheredParameters(
                list(base_model.parameters()) + list(target_model.parameters()),
                modifier_rank=0
            ):
                if deepspeed.comm.get_rank() == 0:
                    SyncRefModelCallback._sync_target_model(base_model, target_model, alpha_value)
        else:
            SyncRefModelCallback._sync_target_model(base_model, target_model, alpha_value)

    def on_step_end(self, args, state, control, **kwargs):
        # 如果到达一定步数，就进行一次参考模型的同步
        model: PreTrainedModel = kwargs['model']
        if self.ref_model is not None and (state.global_step % args.ref_model_sync_steps == 0):
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model(model, self.ref_model, args.ref_model_mixup_alpha)

def _win_rate_completions_df(
    state: TrainerState,
    prompts: List[str],
    completions: List[str],
    winner_indices: List[str]
) -> pd.DataFrame:
    current_step = str(state.global_step)
    repeated_steps = [current_step] * len(prompts)
    combined_data = list(zip(repeated_steps, prompts, completions, winner_indices))
    # 将completions中的两部分内容分开
    data_for_frame = [
        (row[0], row[1], row[2][0], row[2][1], row[3])
        for row in combined_data
    ]
    return pd.DataFrame(
        data_for_frame,
        columns=['step', 'prompt', 'reference_model', 'policy', 'winner_index']
    )
