from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = field(default=None, metadata={'help': 'Model checkpoint for weights initialization.'})
    model_revision: str = field(default='main', metadata={'help': 'Specific model version to use. It can be a branch name, a tag name, or a commit id.'})
    torch_dtype: Optional[str] = field(default=None, metadata={'help': 'Override the default `torch.dtype` and load the model under this dtype.', 'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    trust_remote_code: bool = field(default=False, metadata={'help': 'Whether to allow for custom models defined on the Hub in their own modeling files. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.'})
    attn_implementation: Optional[str] = field(default=None, metadata={'help': 'Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`.'})
    use_peft: bool = field(default=False, metadata={'help': 'Whether to use PEFT for training.'})
    lora_r: int = field(default=16, metadata={'help': 'LoRA R value.'})
    lora_alpha: int = field(default=32, metadata={'help': 'LoRA alpha.'})
    lora_dropout: float = field(default=0.05, metadata={'help': 'LoRA dropout.'})
    lora_target_modules: Optional[list[str]] = field(default=None, metadata={'help': 'LoRA target modules.'})
    lora_modules_to_save: Optional[list[str]] = field(default=None, metadata={'help': 'Model layers to unfreeze & train.'})
    lora_task_type: str = field(default='CAUSAL_LM', metadata={'help': "Task type to pass for LoRA (use 'SEQ_CLS' for reward modeling)."})
    use_rslora: bool = field(default=False, metadata={'help': 'Whether to use Rank-Stabilized LoRA, which sets the adapter scaling factor to `lora_alpha/√r`, instead of the original default value of `lora_alpha/r`.'})
    load_in_8bit: bool = field(default=False, metadata={'help': 'Whether to use 8 bit precision for the base model. Works only with LoRA.'})
    load_in_4bit: bool = field(default=False, metadata={'help': 'Whether to use 4 bit precision for the base model. Works only with LoRA.'})
    bnb_4bit_quant_type: str = field(default='nf4', metadata={'help': 'Quantization type.', 'choices': ['fp4', 'nf4']})
    use_bnb_nested_quant: bool = field(default=False, metadata={'help': 'Whether to use nested quantization.'})

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")
        if hasattr(self.lora_target_modules, '__len__') and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]