# Value-Guided Search

This is the official codebase for the paper “[Value‑Guided Search for Efficient Chain‑of‑Thought Reasoning](https://arxiv.org/abs/2505.17373).”

**Datasets.** We release the two datasets described in Section 2.2 of the paper.
They are available on Hugging Face:
1. [`VGS-AI/OpenR1-Cleaned`](https://huggingface.co/datasets/VGS-AI/OpenR1-Cleaned)
2. [`VGS-AI/OpenR1-VM`](https://huggingface.co/datasets/VGS-AI/OpenR1-VM)

**Models.** We release our 1.5B value model which was trained on DeepSeek CoTs in `OpenR1-VM` following the method described in Section 2.1 of the paper.
This model is available on Hugging Face at [`VGS-AI/DeepSeek-VM-1.5B`](https://huggingface.co/VGS-AI/DeepSeek-VM-1.5B).
The model is a `Qwen2ForClassifier` model (custom defined in `classifier_lib`), which is a modified version of the Qwen2 model for classification tasks.

To load the value model, you can use the following code snippet:

```python

import classifier_lib

model_loading_kwargs = dict(attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, use_cache=False)
classifier = classifier_lib.Qwen2ForClassifier.from_pretrained("VGS-AI/DeepSeek-VM-1.5B", **model_loading_kwargs)
```

To apply the model to `input_ids`, you can use the following code snippet:

```python
import torch

device = torch.device("cuda")
# your input_ids
input_ids = torch.tensor([151646, 151644, 18, 13, 47238, ...], dtype=torch.long, device=device)
attention_mask = torch.ones_like(input_ids)
classifier_outputs = classifier(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
# use last index of the sequence
scores = classifier_outputs.success_probs.squeeze(0)[-1].item()
```

## Beam Search with Value Model

`inference_eval.py` contains the code for block-wise beam search with value model. We use SGLang as the inference engine for generating blocks. We use Neptune Scale for logging results. Below is an example command for running value-guided search (VGS) on AIME-24, with beam width 2 (beam2), DVTS (num_repetitions) 16 and beam width (num_blocks) 8, for a total inference budget of 128.

```
python -u inference_eval.py  \
    --benchmark aime-24 \
    --piref_gpu_util 0.75 \
    --piref_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --seed 7996 \
    --batch_size 10 \
    --num_blocks 8 \
    --block_size 4096 \
    --temperature 0.6 \
    --classifier_ckpt_path VGS-AI/DeepSeek-VM-1.5B \
    --num_repetitions 16 \
    --output_path inference_outputs.jsonl \
    --attention_impl flash_attention_2 \
    --search_type beam2 \
    --neptune_project cornell-rl/oss-infer-eval
```

## Train Value Model
`train_classifier.py` contains the code for training the value model. Our code supports multi-GPU and multi-node training. Below is an example script for training a value model on a single node with 4 GPUs. We enable `HF_HUB_ENABLE_HF_TRANSFER` for faster data transfer from HuggingFace.

```
export HF_HUB_ENABLE_HF_TRANSFER=1
torchrun --standalone --nproc_per_node=4 train_classifier.py \
    --eval_every -1 \
    --save_every 486 \
    --num_steps 12170 \
    --max_lr 0.0001 \
    --data_path VGS-AI/OpenR1-VM \
    --data_num_response 56 \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --total_batch_size 64 \
    --micro_batch_size 8 \
    --small_group_size 2 \
    --run_name single_node_run \
    --track \
    --compile \
    --push_to_hub
```
We remark that `DeepSeek-VM-1.5B` was trained on 16 nodes with 8 H100 GPUs, with a `total_batch_size` of 1024, `micro_batch_size` of 8 and no gradient accumulation. Please see Appendix E of the paper for our training hyperparameters.

## Citation
If you find our code or datasets helpful, please consider citing our paper:
```
@article{wang2025value,
  title={Value-Guided Search for Efficient Chain-of-Thought Reasoning},
  author={Wang, Kaiwen and Zhou, Jin Peng and Chang, Jonathan and Gao, Zhaolin and Kallus, Nathan and Brantley, Kiant{\'e} and Sun, Wen},
  journal={arXiv preprint arXiv:2505.17373},
  year={2025}
}
```
