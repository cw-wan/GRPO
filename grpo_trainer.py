import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from import_utils import is_vllm_available
from models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from callbacks import SyncRefModelCallback
from grpo_config import GRPOConfig
from utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

if is_peft_available():
    from peft import PeftConfig, get_peft_model
if is_vllm_available():
    from vllm import LLM, SamplingParams
if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        # Reorganized constructor for a different look
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # Using a flattened list comprehension in a slightly altered way
        all_indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        repeated = [idx for idx in all_indexes for _ in range(self.repeat_count)]
        return iter(repeated)

    def __len__(self):
        return self.num_samples * self.repeat_count


class GRPOTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Provide default GRPOConfig if needed
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            possible_dtype = init_kwargs.get("torch_dtype")
            if possible_dtype is None or possible_dtype == "auto" or isinstance(possible_dtype, torch.dtype):
                pass
            elif isinstance(possible_dtype, str):
                init_kwargs["torch_dtype"] = getattr(torch, possible_dtype)
            else:
                raise ValueError(
                    f"传递给 `GRPOConfig` 的 `torch_dtype` 无效。期望值为 'auto' 或表示 `torch.dtype` 的字符串（例如 'float32'），但接收到 {possible_dtype}."
                )
            if args.gradient_checkpointing:
                init_kwargs["use_cache"] = False if "use_cache" not in init_kwargs else init_kwargs.get("use_cache")
            model = AutoModelForCausalLM.from_pretrained(model, **init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "您向 `GRPOConfig` 传入了 `model_init_kwargs`，但您的模型已经被实例化。仅当 `model` 参数是一个字符串时才能使用这个参数。"
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **init_kwargs)
        elif not is_peft_model(model):
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        for idx_func, rfunc in enumerate(reward_funcs):
            if isinstance(rfunc, str):
                reward_funcs[idx_func] = AutoModelForSequenceClassification.from_pretrained(
                    rfunc, num_labels=1, **init_kwargs
                )

        self.reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"奖励权重的数量({len(args.reward_weights)})必须与奖励函数的数量({len(reward_funcs)})相匹配。"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        elif len(reward_processing_classes) != len(reward_funcs):
            raise ValueError("奖励处理类的数量必须与奖励函数的数量相匹配。")

        for idx_cls, (processing_cls, rfunc) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(rfunc, PreTrainedModel):
                if processing_cls is None:
                    processing_cls = AutoTokenizer.from_pretrained(rfunc.config._name_or_path)
                if processing_cls.pad_token_id is None:
                    processing_cls.pad_token = processing_cls.eos_token
                rfunc.config.pad_token_id = processing_cls.pad_token_id
                reward_processing_classes[idx_cls] = processing_cls

        self.reward_processing_classes = reward_processing_classes

        def local_data_collator(f_list):
            return f_list

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_vllm = args.use_vllm
        self.beta = args.beta
        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=local_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Validate the batch sizes wrt num_generations
        world_size = self.accelerator.num_processes
        train_bs = args.per_device_train_batch_size * world_size
        valid_n_gens = [
            n_g for n_g in range(2, train_bs + 1) if train_bs % n_g == 0
        ]
        if self.num_generations not in valid_n_gens:
            raise ValueError(
                f"全局训练批大小({world_size} x {args.per_device_train_batch_size})必须能被每个提示生成次数({self.num_generations})整除。"
                f"基于当前的训练批大小，可用的有效生成次数为: {valid_n_gens}."
            )

        if self.args.eval_strategy != "no":
            eval_bs = args.per_device_eval_batch_size * world_size
            possible_eval_gens = [
                n_g for n_g in range(2, eval_bs + 1) if eval_bs % n_g == 0
            ]
            if self.num_generations not in possible_eval_gens:
                raise ValueError(
                    f"全局评估批大小({world_size} x {args.per_device_eval_batch_size})必须能被每个提示生成次数({self.num_generations})整除。"
                    f"基于当前的评估批大小，可用的有效生成次数为: {possible_eval_gens}."
                )

        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM 不可用且 `use_vllm` 被设置为 True。请通过 `pip install vllm` 安装 vLLM 才能使用它。"
                )
            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"
                parts = vllm_device.split(":")
                if parts[0] == "cuda" and int(parts[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"为 vllm 请求的设备({vllm_device})不可用。您可能在未限制训练所用 GPU 数量的情况下使用了 vLLM。"
                        f"将 `--num_processes` 参数设置为小于您机器上可用的 GPU 数量，一般减少一即可满足要求。例如: "
                        f"`--num_processes {torch.cuda.device_count() - 1}`."
                    )
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"警告：请求的设备 {vllm_device} 同样被用于训练。为了获得更高的吞吐量并避免内存不足错误，"
                        "建议为 vLLM 使用一个专用设备。如果这是有意而为，可以忽略此警告，但应适当调整 "
                        "`vllm_gpu_memory_utilization`。"
                    )
                # Preserve logic for patch usage
                with patch("torch.distributed.get_world_size", return_value=1), patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                ):
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature, max_tokens=self.max_completion_length
                )
            self._last_loaded_step = 0
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        self.model_accepts_loss_kwargs = False
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for idx_func, rfunc in enumerate(self.reward_funcs):
            if isinstance(rfunc, PreTrainedModel):
                self.reward_funcs[idx_func] = self.accelerator.prepare_model(rfunc, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # A slight restructuring of logic
        results = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        truncated = results[:, :-1, :]
        last_ids = input_ids[:, -logits_to_keep:]
        relevant = truncated[:, -logits_to_keep:]
        return selective_log_softmax(relevant, last_ids)

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped:
            real_model = unwrapped
            if is_compiled_module(unwrapped):
                real_model = unwrapped._orig_mod
            if is_peft_model(real_model):
                real_model.merge_adapter()
                net_sd = real_model.state_dict()
                net_sd = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v
                    for k, v in net_sd.items()
                }
                net_sd = {
                    k: v
                    for k, v in net_sd.items()
                    if real_model.prefix not in k
                }
                net_sd = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in net_sd.items()
                    if "original_module" not in k
                }
            else:
                net_sd = real_model.state_dict()

            if self.accelerator.is_main_process:
                worker_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                worker_model.load_weights(net_sd.items())

            if is_peft_model(real_model):
                real_model.unmerge_adapter()

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # Restructuring logic while retaining the exact function
        device_in_use = self.accelerator.device
        all_prompts = [obj["prompt"] for obj in inputs]
        template_prompts = [
            maybe_apply_chat_template(x, self.processing_class)["prompt"] for x in inputs
        ]
        tokenized = self.processing_class(
            template_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        tokenized = super()._prepare_inputs(tokenized)
        p_ids, p_mask = tokenized["input_ids"], tokenized["attention_mask"]

        if self.max_prompt_length is not None:
            p_ids = p_ids[:, -self.max_prompt_length :]
            p_mask = p_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            gathered_prompts = gather_object(template_prompts)
            if self.accelerator.is_main_process:
                generated = self.llm.generate(
                    gathered_prompts, sampling_params=self.sampling_params, use_tqdm=False
                )
                comp_ids = [o.token_ids for group in generated for o in group.outputs]
            else:
                comp_ids = [None] * len(gathered_prompts)
            comp_ids = broadcast_object_list(comp_ids, from_process=0)
            begin = self.accelerator.process_index * len(all_prompts)
            finish = (self.accelerator.process_index + 1) * len(all_prompts)
            sub_ids = comp_ids[begin:finish]
            comp_tensors = [torch.tensor(g, device=device_in_use) for g in sub_ids]
            padded_comp = pad(comp_tensors, padding_value=self.processing_class.pad_token_id)
            merged_ids = torch.cat([p_ids, padded_comp], dim=1)
        else:
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped:
                merged_ids = unwrapped.generate(
                    p_ids,
                    attention_mask=p_mask,
                    generation_config=self.generation_config,
                )
            plength = p_ids.size(1)
            p_ids = merged_ids[:, :plength]
            comp_tensors = merged_ids[:, plength:]

        # Detect early termination from EOS
        eos_flags = comp_tensors == self.processing_class.eos_token_id
        eos_positions = torch.full(
            (eos_flags.size(0),), eos_flags.size(1), dtype=torch.long, device=device_in_use
        )
        any_eos = eos_flags.any(dim=1)
        eos_positions[any_eos] = eos_flags.int().argmax(dim=1)[any_eos]
        seq_range = torch.arange(eos_flags.size(1), device=device_in_use).expand(eos_flags.size(0), -1)
        comp_mask = (seq_range <= eos_positions.unsqueeze(1)).int()
        combined_mask = torch.cat([p_mask, comp_mask], dim=1)

        # Gather needed logprobs
        keep_count = comp_tensors.size(1)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_logps = self._get_per_token_logps(
                    self.ref_model,
                    merged_ids,
                    combined_mask,
                    keep_count,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_logps = self._get_per_token_logps(
                        self.model,
                        merged_ids,
                        combined_mask,
                        keep_count,
                    )

        # Decode completions
        decoded_comps = self.processing_class.batch_decode(comp_tensors, skip_special_tokens=True)

        # Reattach completions in conversational style if needed
        if is_conversational(inputs[0]):
            final_completions = []
            for p_struct, c_text in zip(all_prompts, decoded_comps):
                if p_struct[-1]["role"] == "assistant":
                    bootstrap = p_struct.pop()["content"]
                else:
                    bootstrap = ""
                final_completions.append([{"role": "assistant", "content": bootstrap + c_text}])
        else:
            final_completions = decoded_comps

        # Calculate rewards
        all_rewards = torch.zeros(len(all_prompts), len(self.reward_funcs), device=device_in_use)
        for idx, (rfunc, rproc_cls) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(rfunc, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [
                        {"messages": a_p + a_c}
                        for a_p, a_c in zip(all_prompts, final_completions)
                    ]
                    text_batch = [
                        apply_chat_template(msg, rproc_cls)["text"]
                        for msg in messages
                    ]
                else:
                    text_batch = [
                        p + c for p, c in zip(all_prompts, final_completions)
                    ]
                reward_inp = rproc_cls(
                    text_batch,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inp = super()._prepare_inputs(reward_inp)
                with torch.inference_mode():
                    all_rewards[:, idx] = rfunc(**reward_inp).logits[:, 0]
            else:
                extra_keys = [k for k in inputs[0] if k not in ["prompt", "completion"]]
                rkwargs = {k: [item[k] for item in inputs] for k in extra_keys}
                func_result = rfunc(prompts=all_prompts, completions=final_completions, **rkwargs)
                all_rewards[:, idx] = torch.tensor(func_result, dtype=torch.float32, device=device_in_use)

        # Combine and standardize
        all_rewards = gather(all_rewards)
        weighted_sum = (all_rewards * self.reward_weights.to(device_in_use).unsqueeze(0)).sum(dim=1)
        grouped_means = weighted_sum.view(-1, self.num_generations).mean(dim=1)
        grouped_stds = weighted_sum.view(-1, self.num_generations).std(dim=1)
        repeated_mean = grouped_means.repeat_interleave(self.num_generations, dim=0)
        repeated_std = grouped_stds.repeat_interleave(self.num_generations, dim=0)
        advantage_vals = (weighted_sum - repeated_mean) / (repeated_std + 0.0001)
        start_slice = self.accelerator.process_index * len(all_prompts)
        stop_slice = (self.accelerator.process_index + 1) * len(all_prompts)
        advantage_vals = advantage_vals[start_slice:stop_slice]

        reward_avg = all_rewards.mean(dim=0)
        for idx, rfunc in enumerate(self.reward_funcs):
            if isinstance(rfunc, nn.Module):
                r_name = rfunc.config._name_or_path.split("/")[-1]
            else:
                r_name = rfunc.__name__
            self._metrics[f"rewards/{r_name}"].append(reward_avg[idx].item())
        self._metrics["reward"].append(weighted_sum.mean().item())
        self._metrics["reward_std"].append(grouped_stds.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and ("wandb" in self.args.report_to)
        ):
            import pandas as pd

            table_dict = {
                "step": [str(self.state.global_step)] * weighted_sum.size(0),
                "prompt": gather_object(template_prompts),
                "completion": gather_object(decoded_comps),
                "reward": weighted_sum.tolist(),
            }
            df_table = pd.DataFrame(table_dict)
            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df_table)})

        return {
            "prompt_ids": p_ids,
            "prompt_mask": p_mask,
            "completion_ids": comp_tensors,
            "completion_mask": comp_mask,
            "ref_per_token_logps": ref_logps,
            "advantages": advantage_vals,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer 不支持返回输出。")

        pid, pmask = inputs["prompt_ids"], inputs["prompt_mask"]
        cid, cmask = inputs["completion_ids"], inputs["completion_mask"]
        joined_ids = torch.cat([pid, cid], dim=1)
        joined_mask = torch.cat([pmask, cmask], dim=1)
        c_length = cid.size(1)
        token_logps = self._get_per_token_logps(model, joined_ids, joined_mask, c_length)
        ref_logps = inputs["ref_per_token_logps"]

        diff_logps = ref_logps - token_logps
        kl_part = torch.exp(diff_logps) - diff_logps - 1
        adv_vals = inputs["advantages"]
        ratio_exp = torch.exp(token_logps - token_logps.detach()) * adv_vals.unsqueeze(1)
        per_token = -(ratio_exp - self.beta * kl_part)
        masked_loss = (per_token * cmask).sum(dim=1) / cmask.sum(dim=1)
        final_loss = masked_loss.mean()

        comp_len_mean = self.accelerator.gather_for_metrics(cmask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(comp_len_mean)

        kl_avg = ((kl_part * cmask).sum(dim=1) / cmask.sum(dim=1)).mean()
        kl_scalar = self.accelerator.gather_for_metrics(kl_avg).mean().item()
        self._metrics["kl"].append(kl_scalar)

        return final_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        ready_inp = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                local_loss = self.compute_loss(model, ready_inp)
            local_loss = local_loss.mean().detach()
        return (local_loss, None, None)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Some re-labelling around logging
        aggregated = {k: sum(v) / len(v) for k, v in self._metrics.items()}
        if next(iter(logs.keys())).startswith("eval_"):
            aggregated = {f"eval_{nm}": val for nm, val in aggregated.items()}
        updated_logs = {**logs, **aggregated}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(updated_logs, start_time)
        else:
            super().log(updated_logs)
        self._metrics.clear()
