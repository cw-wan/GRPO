import re
from datasets import load_dataset

def _load_dataset(name, subset, split, ratio):
    return load_dataset(name, subset, split=f'{split}[:{ratio}%]')

def _batch_parse_answers(ans):
    pattern = '\\d+(?:/\\d+|\\.\\d+|(?:,\\d+)+)?'
    results = []
    for text in ans:
        numbers = re.findall(pattern, text)
        results.append(numbers[-1].replace(',', '') if numbers else None)
    return results