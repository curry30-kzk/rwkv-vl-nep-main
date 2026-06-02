# RWKV-NEPA Full ImageNet Pretraining Results

## Experiment

Formal full ImageNet-1K RWKV-NEPA pretraining.

- Model: RWKV-NEPA
- Dataset: full ImageNet-1K HuggingFace Arrow dataset
- GPUs: 4 × A800
- Global batch size: 4096
- Per-device batch size: 64
- Gradient accumulation steps: 16
- Precision: FP32 / TF32
- Optimizer: adamw_torch_fused
- LR: 4.8e-3
- Scheduler: cosine
- Warmup ratio: 0.025
- Weight decay: 0.05
- WKV: fused CUDA WKV kernel enabled
- Planned protocol: 1600 epochs

## Main result directory on server

`/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/outputs_pretrain/rwkv_nepa_full_pretrain_ep1600_bs4096_fp32tf32_wkv_b64acc16_4gpu_seed42`

## Important checkpoints

### 70ep checkpoint

`checkpoint-21910`

- global_step: 21910
- approximate epoch: 70
- raw loss was previously checked and stayed around -0.93 to -0.95.

### 80ep checkpoint

`checkpoint-25040`

- global_step: 25040
- approximate epoch: 80
- file integrity checked
- raw outputs.loss checked: approximately -0.91 to -0.93
- no NaN/Inf found.

### 100ep verified checkpoint

`checkpoint-31300_ep100_rawloss_verified`

- global_step: 31300
- approximate epoch: 100
- copied from the previous suspect directory after verification
- raw outputs.loss checked: approximately -0.92 to -0.94
- no NaN/Inf found.
- This is the checkpoint selected for downstream fine-tuning.

## Notes

Trainer outer `loss` logs after resume showed abnormal values such as values close to 0. However, direct `compute_loss` raw `outputs.loss` checks showed that the model loss was stable and finite. Therefore, the outer Trainer loss was treated as a logging/aggregation issue, and raw `outputs.loss` was used for checkpoint health verification.

Large checkpoint weights are not committed to GitHub. See `checkpoint_manifest.txt` and `checkpoint_file_list.txt` for server paths and SHA256 checksums.
