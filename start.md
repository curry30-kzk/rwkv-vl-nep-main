PROJECT_DIR=/public/home/ssjxkzk/rwkv-vl-nep-main/rwkv-vl-nep-main

KERNEL_PKG_DIR=${PROJECT_DIR}/.kernel_pkg_test/kernels_0_11_0
KERNEL_HF_HOME=${PROJECT_DIR}/.hf_cache_kernel_test
TORCH_EXT_DIR=${PROJECT_DIR}/.torch_extensions_rwkv

FULL_DATASET=/public/home/ssjxkzk/rwkv-vl-nep-main/data/imagenet-1k-hf
CONFIG_DIR=${PROJECT_DIR}/outputs/rwkv_nepa_transfer_main_bs8

OUT_DIR=${PROJECT_DIR}/outputs_pretrain/rwkv_nepa_full_pretrain_ep1600_bs4096_fp32tf32_wkv_b64acc16_4gpu_seed42_from_scratch_$(date +%Y%m%d_%H%M%S)

mkdir -p ${OUT_DIR}

LOG=${OUT_DIR}/pretrain_from_scratch_ep1600_bs4096_4gpu_$(date +%Y%m%d_%H%M%S).log

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=4 \
NCCL_DEBUG=WARN \
TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 \
TORCH_NCCL_DUMP_ON_TIMEOUT=1 \
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
PYTHONPATH=${KERNEL_PKG_DIR}:${PYTHONPATH} \
HF_HOME=${KERNEL_HF_HOME} \
HF_DATASETS_CACHE=${PROJECT_DIR}/hf_cache \
TORCH_EXTENSIONS_DIR=${TORCH_EXT_DIR} \
TORCH_CUDA_ARCH_LIST="8.0" \
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u NEPA_NAN_DEBUG \
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
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1600 \
  --learning_rate 4.8e-3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.025 \
  --weight_decay 0.05 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --optim adamw_torch_fused \
  --logging_steps 20 \
  --save_strategy steps \
  --save_steps 3130 \
  --save_only_model False \
  --save_total_limit 20 \
  --dataloader_num_workers 8 \
  --dataloader_prefetch_factor 4 \
  --dataloader_persistent_workers True \
  --remove_unused_columns False \
  --bf16 False \
  --fp16 False \
  --tf32 True \
  --ddp_find_unused_parameters True \
  --report_to none \
  2>&1 | tee -a ${LOG}