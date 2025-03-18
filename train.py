import json
from grpo_config import GRPOConfig
from grpo_trainer import GRPOTrainer
from QWen_utils import _load_dataset, _batch_parse_answers

MODELS = {'Qwen2.5-0.5B-Instruct': 'Qwen/Qwen2.5-0.5B-Instruct'}
DATASETS = {'GSM8K': lambda ratio: _load_dataset('openai/gsm8k', 'main', split='train', ratio=ratio)}
CONVERTION = {'GSM8K': lambda ds: ds.rename_columns({'question': 'prompt', 'answer': 'completion'})}

with open('config.json', 'r') as cfg_file:
    config = json.load(cfg_file)

model_name = config['model']
dataset_name = config['dataset']
train_ratio = config['train_ratio']

train_data = DATASETS[dataset_name](train_ratio)
train_data = CONVERTION[dataset_name](train_data)
train_data = train_data.map(lambda x: {'ground_truth': _batch_parse_answers([x['completion']])[0]})

print(train_data[0])

def reward_func(completions, ground_truth, **kwargs):
    prd, gt_ = _batch_parse_answers(completions), ground_truth
    return [1.0 if a == b else 0.0 for a, b in zip(prd, gt_)]

training_args = GRPOConfig(output_dir=f"{config['model']}-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model=MODELS[model_name],
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=train_data
)
trainer.train()
