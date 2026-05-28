#!/usr/bin/env bash

cd /public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main

KERNEL_PKG_DIR=/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/.kernel_pkg_test/kernels_0_11_0
KERNEL_HF_HOME=/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/.hf_cache_kernel_test
TORCH_EXT_DIR=/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/.torch_extensions_rwkv

FULL_DATASET=/public/home/ssjxkzk/rwkv-vl-nep-main/data/imagenet-1k-hf
CONFIG_DIR=/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/outputs/rwkv_nepa_transfer_main_bs8
OUT_BASE=/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/outputs_pretrain/speedtests_fp32_wkv

mkdir -p ${OUT_BASE}

run_test () {
  NAME=$1
  PER_DEVICE_BS=$2
  GRAD_ACCUM=$3
  DDP_UNUSED=$4
  MAX_STEPS=$5

  OUT_DIR=${OUT_BASE}/${NAME}
  mkdir -p ${OUT_DIR}

  echo "============================================================"
  echo "Running ${NAME}"
  echo "per_device_train_batch_size=${PER_DEVICE_BS}"
  echo "gradient_accumulation_steps=${GRAD_ACCUM}"
  echo "global_batch=$((4 * PER_DEVICE_BS * GRAD_ACCUM))"
  echo "ddp_find_unused_parameters=${DDP_UNUSED}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OMP_NUM_THREADS=4 \
  NCCL_DEBUG=WARN \
  TORCH_FR_BUFFER_SIZE=1048576 \
  TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
  PYTHONPATH=${KERNEL_PKG_DIR}:${PYTHONPATH} \
  HF_HOME=${KERNEL_HF_HOME} \
  HF_DATASETS_CACHE=/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main/hf_cache \
  TORCH_EXTENSIONS_DIR=${TORCH_EXT_DIR} \
  TORCH_CUDA_ARCH_LIST="8.0" \
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
  torchrun --nproc_per_node=4 run_nepa1.py \
    --model_type rwkv_nepa \
    --config_name ${CONFIG_DIR} \
    --image_processor_name ${CONFIG_DIR} \
    --dataset_name ${FULL_DATASET} \
    --load_from_disk True \
    --do_train \
    --output_dir ${OUT_DIR} \
    --overwrite_output_dir True \
    --seed 42 \
    --data_seed 42 \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --max_steps ${MAX_STEPS} \
    --learning_rate 4.8e-3 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.025 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --logging_steps 5 \
    --save_strategy no \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 4 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --bf16 False \
    --fp16 False \
    --tf32 True \
    --ddp_find_unused_parameters ${DDP_UNUSED} \
    --report_to none \
    2>&1 | tee ${OUT_DIR}/speedtest.log

  echo "-------------------- RESULT: ${NAME} --------------------"
  grep -E "Loading CUDA kernel|train_runtime|train_samples_per_second|train_steps_per_second|train_loss|loss|grad_norm|CUDA out of memory|RuntimeError|unused" \
    ${OUT_DIR}/speedtest.log || true
  echo
}

# A: 增大单卡 batch，减少梯度累积次数
# 4 × 128 × 8 = 4096
run_test A_b128_acc8_ddpTrue 128 8 True 20

# B: 在 A 的基础上关闭 DDP unused parameter 检查
# 如果不报错，通常更快
run_test B_b128_acc8_ddpFalse 128 8 False 20

echo "============================================================"
echo "All speed tests finished. Logs are under:"
echo "${OUT_BASE}"
echo "============================================================"

