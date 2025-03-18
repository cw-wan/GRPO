from typing import Any, Callable, Optional, Sequence, TypeVar, Union
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

DatasetType = TypeVar('DatasetType', Dataset, DatasetDict)

def is_conversational(example: dict[str, Any]) -> bool:
    possible_keys = {'prompt', 'chosen', 'rejected', 'completion', 'messages'}
    found = {k for k in example if k in possible_keys}
    if not found:
        return False
    one_key = found.pop()
    content = example[one_key]
    if isinstance(content, list):
        first_item = content[0]
        if isinstance(first_item, dict) and 'role' in first_item and 'content' in first_item:
            return True
    return False

def apply_chat_template(example: dict[str, list[dict[str, str]]],
                        tokenizer: PreTrainedTokenizer,
                        tools: Optional[list[Union[dict, Callable]]] = None) -> dict[str, str]:
    check_set = {'prompt', 'chosen', 'rejected', 'completion', 'messages', 'label'}
    got_keys = {key for key in example if key in check_set}
    valid_combinations = [
        {'messages'},
        {'prompt'},
        {'prompt', 'completion'},
        {'prompt', 'chosen', 'rejected'},
        {'chosen', 'rejected'},
        {'prompt', 'completion', 'label'}
    ]
    if got_keys not in valid_combinations:
        raise KeyError(f'在该示例中存在无效的键: {got_keys}')
    messages = None
    prompt = None
    chosen = None
    rejected = None
    completion = None
    if 'messages' in example:
        messages = tokenizer.apply_chat_template(example['messages'], tools=tools, tokenize=False)
    if 'prompt' in example:
        last_role = example['prompt'][-1]['role']
        if last_role == 'user':
            add_gen = True
            cont_msg = False
        elif last_role == 'assistant':
            add_gen = False
            cont_msg = True
        else:
            raise ValueError(f'最后消息中的角色无效: {last_role}')
        prompt = tokenizer.apply_chat_template(
            example['prompt'],
            tools=tools,
            continue_final_message=cont_msg,
            tokenize=False,
            add_generation_prompt=add_gen
        )
        if 'chosen' in example:
            merged_chosen = tokenizer.apply_chat_template(example['prompt'] + example['chosen'], tools=tools, tokenize=False)
            chosen = merged_chosen[len(prompt):]
        if 'rejected' in example:
            merged_rejected = tokenizer.apply_chat_template(example['prompt'] + example['rejected'], tools=tools, tokenize=False)
            rejected = merged_rejected[len(prompt):]
        if 'completion' in example:
            merged_completion = tokenizer.apply_chat_template(example['prompt'] + example['completion'], tools=tools, tokenize=False)
            completion = merged_completion[len(prompt):]
    else:
        if 'chosen' in example:
            chosen = tokenizer.apply_chat_template(example['chosen'], tools=tools, tokenize=False)
        if 'rejected' in example:
            rejected = tokenizer.apply_chat_template(example['rejected'], tools=tools, tokenize=False)
    if 'prompt' in example:
        note = (
            '应用于 prompt + completion 的聊天模板并未以仅应用于 prompt 的聊天模板开头。\n'
            '此聊天模板不被 TRL 支持。\n'
            '**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}'
        )
        if 'chosen' in example and chosen is not None:
            if not (prompt is not None and merged_chosen.startswith(prompt)):
                raise ValueError(note.format(prompt, merged_chosen))
        if 'rejected' in example and rejected is not None:
            if not (prompt is not None and merged_rejected.startswith(prompt)):
                raise ValueError(note.format(prompt, merged_rejected))
        if 'completion' in example and completion is not None:
            if not (prompt is not None and merged_completion.startswith(prompt)):
                raise ValueError(note.format(prompt, merged_completion))
    outcome = {}
    if messages is not None:
        outcome['text'] = messages
    if prompt is not None:
        outcome['prompt'] = prompt
    if chosen is not None:
        outcome['chosen'] = chosen
    if rejected is not None:
        outcome['rejected'] = rejected
    if completion is not None:
        outcome['completion'] = completion
    if 'label' in example:
        outcome['label'] = example['label']
    return outcome

def maybe_apply_chat_template(example: dict[str, list[dict[str, str]]],
                              tokenizer: PreTrainedTokenizer,
                              tools: Optional[list[Union[dict, Callable]]] = None) -> dict[str, str]:
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
    return example

def _unpair_row(examples: list[dict[str, list[dict[str, str]]]]) -> list[dict[str, list[dict[str, str]]]]:
    count = len(examples['chosen'])
    paired = {
        'completion': examples['chosen'] + examples['rejected'],
        'label': [True] * count + [False] * count
    }
    if 'prompt' in examples:
        paired['prompt'] = examples['prompt'] + examples['prompt']
    return paired

def unpair_preference_dataset(dataset: DatasetType,
                              num_proc: Optional[int] = None,
                              desc: Optional[str] = None) -> DatasetType:
    return dataset.map(
        _unpair_row,
        batched=True,
        remove_columns=['chosen', 'rejected'],
        num_proc=num_proc,
        desc=desc
    )

def extract_prompt(example: dict[str, Sequence]) -> dict[str, Sequence]:
    shorter = min(len(example['chosen']), len(example['rejected']))
    index = 0
    for i in range(shorter):
        if example['chosen'][i] != example['rejected'][i]:
            if example['chosen'][i - 1] == ' ':
                i -= 1
            index = i
            break
    return {
        'prompt': example['chosen'][:index],
        'chosen': example['chosen'][index:],
        'rejected': example['rejected'][index:]
    }
