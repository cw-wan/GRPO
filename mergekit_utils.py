import torch
from huggingface_hub import HfApi
from trl.import_utils import is_mergekit_available
if is_mergekit_available():
    from mergekit.config import MergeConfiguration
    from mergekit.merge import MergeOptions, run_merge

def upload_model_to_hf(folder_path: str, repo_id: str):
    api_instance = HfApi()
    repository = api_instance.create_repo(repo_id, repo_type="model")
    api_instance.upload_folder(
        folder_path=folder_path,
        repo_id=repository.repo_id,
        repo_type=repository.repo_type
    )

class MergeConfig:
    def __init__(self, method: str = "linear"):
        if not is_mergekit_available():
            raise ImportError("MergeConfig 需要 `mergekit` 附加组件。要安装，请运行 `pip install trl[mergekit]`.")
        self.method = method
        self.policy_model_path = None
        self.target_model_path = None
        if method == "linear":
            self.policy_model_weight = 0.5
            self.target_model_weight = 0.5
            self.dtype = "float16"
        elif method == "ties":
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = "float16"
        elif method == "dare_ties":
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = "float16"
        elif method == "slerp":
            self.t_values = 0.5
            self.dtype = "float16"
        else:
            raise ValueError(f"不支持的合并方法: {method}")

    def create_merge_config_linear(self) -> "MergeConfiguration":
        data = {
            "dtype": self.dtype,
            "merge_method": "linear",
            "models": [
                {"model": self.policy_model_path, "parameters": {"weight": self.policy_model_weight}},
                {"model": self.target_model_path, "parameters": {"weight": self.target_model_weight}}
            ],
        }
        return MergeConfiguration.model_validate(data)

    def create_merge_config_ties(self) -> "MergeConfiguration":
        data = {
            "merge_method": "ties",
            "slices": None,
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None
                    },
                    "parameters": {"density": self.target_model_density, "weight": self.target_model_weight},
                },
                {
                    "model": {
                        "model": {"path": self.policy_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None
                    },
                    "parameters": {"density": self.policy_model_density, "weight": self.policy_model_weight},
                }
            ],
            "parameters": {"normalize": self.normalize},
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None
        }
        return MergeConfiguration.model_validate(data)

    def create_merge_config_dare_ties(self) -> "MergeConfiguration":
        data = {
            "merge_method": "dare_ties",
            "slices": None,
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None
                    },
                    "parameters": {"density": self.target_model_density, "weight": self.target_model_weight},
                },
                {
                    "model": {
                        "model": {"path": self.policy_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None
                    },
                    "parameters": {"density": self.policy_model_density, "weight": self.policy_model_weight},
                }
            ],
            "parameters": {"normalize": self.normalize},
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None
        }
        return MergeConfiguration.model_validate(data)

    def create_merge_config_slerp(self) -> "MergeConfiguration":
        data = {
            "merge_method": "slerp",
            "slices": None,
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None
                    },
                    "parameters": None
                }
            ],
            "parameters": {"t": self.t_values},
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None
        }
        return MergeConfiguration.model_validate(data)

    def create(self) -> "MergeConfiguration":
        if self.method == "linear":
            return self.create_merge_config_linear()
        elif self.method == "ties":
            return self.create_merge_config_ties()
        elif self.method == "dare_ties":
            return self.create_merge_config_dare_ties()
        elif self.method == "slerp":
            return self.create_merge_config_slerp()

def merge_models(config: MergeConfig, out_path: str):
    if not is_mergekit_available():
        raise ImportError("merge_models 需要 `mergekit` 附加组件。要安装，请运行 `pip install trl[mergekit]`.")
    run_merge(
        config,
        out_path=out_path,
        options=MergeOptions(
            cuda=torch.cuda.is_available(),
            copy_tokenizer=True,
            lazy_unpickle=False,
            low_cpu_memory=False
        ),
    )
