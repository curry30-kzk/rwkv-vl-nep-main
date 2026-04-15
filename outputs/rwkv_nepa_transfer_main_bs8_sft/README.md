---
library_name: transformers
base_model: ./outputs/rwkv_nepa_transfer_main_bs8_sft/checkpoint-53382
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagenet-1k
metrics:
- accuracy
model-index:
- name: rwkv_nepa_transfer_main_bs8_sft
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: /public/home/ssjxkzk/rwkv-vl-nep-main/data/imagenet-1k-hf
      type: imagenet-1k
      config: default
      split: validation
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.59176
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# rwkv_nepa_transfer_main_bs8_sft

This model is a fine-tuned version of [./outputs/rwkv_nepa_transfer_main_bs8_sft/checkpoint-53382](https://huggingface.co/./outputs/rwkv_nepa_transfer_main_bs8_sft/checkpoint-53382) on the /public/home/ssjxkzk/rwkv-vl-nep-main/data/imagenet-1k-hf dataset.
It achieves the following results on the evaluation set:
- Loss: 1.8891
- Accuracy: 0.5918

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.006
- train_batch_size: 16
- eval_batch_size: 32
- seed: 42
- distributed_type: multi-GPU
- num_devices: 3
- gradient_accumulation_steps: 3
- total_train_batch_size: 144
- total_eval_batch_size: 96
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.2
- num_epochs: 10.0

### Training results

| Training Loss | Epoch | Step  | Accuracy | Validation Loss |
|:-------------:|:-----:|:-----:|:--------:|:---------------:|
| 5.943         | 1.0   | 8008  | 0.1871   | 4.1448          |
| 5.652         | 2.0   | 16016 | 0.2786   | 3.6050          |
| 5.4447        | 3.0   | 24024 | 0.3496   | 3.1665          |
| 5.2445        | 3.0   | 26691 | 0.3793   | 2.9831          |
| 5.0297        | 4.0   | 35588 | 0.4607   | 2.5575          |
| 4.7887        | 5.0   | 44485 | 0.5220   | 2.2411          |
| 4.7332        | 7.0   | 62279 | 2.2851   | 0.5115          |
| 4.6369        | 8.0   | 71176 | 2.0810   | 0.5536          |
| 4.4834        | 9.0   | 80073 | 1.9236   | 0.5854          |
| 4.5918        | 10.0  | 88970 | 1.8891   | 0.5918          |


### Framework versions

- Transformers 4.56.2
- Pytorch 2.8.0+cu128
- Datasets 3.6.0
- Tokenizers 0.22.2
