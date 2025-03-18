import argparse
import json
from tqdm import tqdm
from models_QWen import *
from QWen_utils import _load_dataset, _batch_parse_answers

MODELS = {
    'Qwen2.5-0.5B-Instruct': Qwen('Qwen/Qwen2.5-0.5B-Instruct'),
    'Qwen2.5-0.5B-Instruct-GRPO': lambda ckpt: Qwen(
        'Qwen/Qwen2.5-0.5B-Instruct', load_checkpoint=True, checkpoint_path=ckpt
    ),
    'Qwen2.5-0.5B-Instruct-SFT': lambda ckpt: Qwen(
        'Qwen/Qwen2.5-0.5B-Instruct', load_checkpoint=True, checkpoint_path=ckpt
    ),
}
DATASETS = {
    'GSM8K': lambda ratio: _load_dataset(
        'openai/gsm8k', 'main', split='test', ratio=ratio
    )
}

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--dataset', choices=DATASETS.keys())
    argp.add_argument('--model', choices=MODELS.keys())
    argp.add_argument('--batch_size', required=False, default=4)
    argp.add_argument('--k', type=int, required=False, default=4)
    argp.add_argument('--test_ratio', type=int, required=False, default=10)
    argp.add_argument('--ckpt', required=False, default='')
    args = argp.parse_args()

    if not args.ckpt:
        current_model = MODELS[args.model]
    else:
        current_model = MODELS[args.model](ckpt=args.ckpt)

    loaded_data = DATASETS[args.dataset](args.test_ratio)
    score_list = []
    batch_total = len(loaded_data) // args.batch_size
    progress_loop = tqdm(
        enumerate(loaded_data.iter(batch_size=args.batch_size)), total=batch_total
    )
    stored_cases = []

    for _, batch in progress_loop:
        progress_loop.set_description('Evaluating {} on {}'.format(args.model, args.dataset))
        correct_values = _batch_parse_answers(batch['answer'])
        repeats = []
        raw_outputs = []
        for _ in range(args.k):
            inference_result = current_model.batch_inference(batch['question'])
            raw_outputs.append(inference_result)
            repeats.append(_batch_parse_answers(inference_result))
        for ques, ans, model_output in zip(
            batch['question'], batch['answer'], list(zip(*raw_outputs))
        ):
            stored_cases.append({
                'Question': ques,
                'Ground Truth': ans,
                'Model Predictions': model_output
            })
            with open(f'{args.model}-cases.json', 'w') as f:
                json.dump(stored_cases, f)
        repeats = list(zip(*repeats))
        for truth, candidate in zip(correct_values, repeats):
            print(truth, candidate)
            match = 0
            for each_pred in candidate:
                if each_pred == truth:
                    match += 1
            score_list.append(match / len(candidate))

    print(
        'Pass@1: {:.2f}%'.format(
            100 * sum(score_list) / len(score_list)
        )
    )
