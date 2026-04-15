---
library_name: transformers
tags:
- embedded-prediction
- vision
- generated_from_trainer
datasets:
- imagenet-1k
model-index:
- name: nepa_base_pilot_5x4090d_retry
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# nepa_base_pilot_5x4090d_retry

This model is a fine-tuned version of [](https://huggingface.co/) on the /public/home/ssjxkzk/rwkv-vl-nep-main/data/imagenet-1k-hf dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0048
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 5
- gradient_accumulation_steps: 2
- total_train_batch_size: 160
- total_eval_batch_size: 40
- optimizer: Use adamw_torch_fused with betas=(0.9,0.95) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.025
- num_epochs: 2.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 4.56.2
- Pytorch 2.8.0+cu128
- Datasets 3.6.0
- Tokenizers 0.22.2
