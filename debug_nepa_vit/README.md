---
library_name: transformers
tags:
- embedded-prediction
- vision
- generated_from_trainer
model-index:
- name: debug_nepa_vit
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# debug_nepa_vit

This model is a fine-tuned version of [](https://huggingface.co/) on the /public/home/ssjxkzk/imagenette_dataset dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1.0

### Training results



### Framework versions

- Transformers 4.56.2
- Pytorch 2.8.0+cu128
- Datasets 3.6.0
- Tokenizers 0.22.2
