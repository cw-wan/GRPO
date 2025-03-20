# Enhancing LLM's Reasoning Ability with Rule-based GRPO

A final project for CSC_52081_EP - Apprentissage Automatique Avancé et Agents Autonomes (2024-2025).

Read our technical report [here](https://github.com/cw-wan/GRPO/blob/main/Report_GRPO.pdf).

## Team Member

Chenwei WAN, Yifei WANG, Hongxu CHEN

## Guide

First adjust necessary configurations in `config.json`, like the ratio of training set to use.

Train GRPO:

```shell
accelerate launch train.py
```

Set evaluation script in `evaluate.sh`:

- Evaluation script for Qwen2.5-0.5B-Instruct:
    ```shell
    python evaluate.py \
    --dataset "GSM8K" \
    --model "Qwen2.5-0.5B-Instruct" \
    --test_ratio 10
    ```
- Evaluation script for Qwen2.5-0.5B-Instruct-GRPO:
    ```shell
    python evaluate.py \
    --dataset "GSM8K" \
    --model "Qwen2.5-0.5B-Instruct-GRPO" \
    --test_ratio 10 \
    --ckpt [checkpoint-path]
    ```
- Evaluation script for Qwen2.5-0.5B-Instruct-SFT:
    ```shell
    python evaluate.py \
    --dataset "GSM8K" \
    --model "Qwen2.5-0.5B-Instruct-SFT" \
    --test_ratio 10 \
    --ckpt [checkpoint-path]
    ```

Run evaluation:

```shell
./evaluate.sh
```

## Results

### GSM8K

| Model                       | Pass@1    |
|-----------------------------|-----------|
| Qwen2.5-0.5B-Instruct-SFT*  | 21.78     |
| Qwen2.5-0.5B-Instruct       | 30.49     |
| Qwen2.5-0.5B-Instruct-GRPO* | **42.61** |

\* Trained on 30% of the training set.

## Reference

[1] Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y.K., Wu, Y. and Guo, D., 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

[2] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X. and Zhang, X., 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.
