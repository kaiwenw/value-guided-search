# Value-Guided Search

This is the official codebase for the paper "Value-Guided Search for Efficient Chain-of-Thought Reasoning".

**Datasets.** We release the two datasets described in Section 2.2 of the paper.
They are available on Hugging Face:
1. [`VGS-AI/OpenR1-Cleaned`](https://huggingface.co/datasets/VGS-AI/OpenR1-Cleaned)
2. [`VGS-AI/OpenR1-VM`](https://huggingface.co/datasets/VGS-AI/OpenR1-VM)

**Models.** We release our 1.5B value model which was trained on DeepSeek CoTs in `OpenR1-VM` following the method described in Section 2.1 of the paper.
This model is available on Hugging Face at [`VGS-AI/DeepSeek-VM-1.5B`](https://huggingface.co/VGS-AI/DeepSeek-VM-1.5B).
The model is a `Qwen2ForClassifier` model (custom defined in `classifier_lib`), which is a modified version of the Qwen2 model for classification tasks.

To load the model, you can use the following code snippet:

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

We will release the training and search inference code soon.