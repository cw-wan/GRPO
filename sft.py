from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from QWen_utils import _load_dataset

train_ds = _load_dataset('openai/gsm8k', 'main', split='train', ratio=30)
train_ds = train_ds.rename_columns({'question': 'prompt', 'answer': 'completion'})
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype='auto', device_map='auto')
training_args = SFTConfig(
    output_dir='Qwen2.5-0.5B-Instruct-SFT',
    logging_steps=10,
    logging_dir='Qwen2.5-0.5B-Instruct-SFT-log',
    max_seq_length=512
)
trainer = SFTTrainer(
    model,
    train_dataset=train_ds,
    args=training_args
)
trainer.train()
