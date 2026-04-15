# ========================
export NCCL_DEBUG=INFO
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export TORCH_DISTRIBUTED_TIMEOUT=1800

: "${WORLD_SIZE:=1}"
: "${RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

# ========================
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

EXPERIMENT_NAME="nepa-base-patch14-224"
WANDB_PROJECT="Nepa-Pretrain"

CONFIG_NAME="configs/pretrain/nepa-base-patch14-224"
DATASET_PATH="data/imagenet-1k-hf"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

TOTAL_BATCH_SIZE=4096
PER_DEVICE_BATCH_SIZE=256
GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU * WORLD_SIZE) ))
NUM_EPOCHS=1600

BASE_LEARNING_RATE=3e-4
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")

DATALOADER_NUM_WORKERS=$((4 * NGPU))

# ========================
# Mixed precision: auto|bf16|fp16|fp32
# - auto picks bf16 if supported, else fp16
MIXED_PRECISION="${MIXED_PRECISION:-auto}"
if [[ "${MIXED_PRECISION}" == "auto" ]]; then
    BF16_OK=$(python - <<'PY'
import torch
supported = False
if torch.cuda.is_available():
    if hasattr(torch.cuda, "is_bf16_supported"):
        supported = torch.cuda.is_bf16_supported()
    else:
        major, _minor = torch.cuda.get_device_capability()
        supported = major >= 8
print("1" if supported else "0")
PY
)
    if [[ "${BF16_OK}" == "1" ]]; then
        MIXED_PRECISION="bf16"
    else
        MIXED_PRECISION="fp16"
    fi
fi

PRECISION_ARGS=()
case "${MIXED_PRECISION}" in
    bf16) PRECISION_ARGS=(--bf16 True) ;;
    fp16) PRECISION_ARGS=(--fp16 True) ;;
    fp32) PRECISION_ARGS=() ;;
    *)
        echo "Unknown MIXED_PRECISION=${MIXED_PRECISION}. Use auto|bf16|fp16|fp32."
        exit 1
        ;;
esac

# ========================
# Optional: enable gradient checkpointing (saves activation memory)
# Usage:
#   USE_GC=1 bash scripts/pretrain/nepa_b.sh
GC_ARGS=()
if [[ "${USE_GC:-0}" == "1" ]]; then
    GC_ARGS=(--gradient_checkpointing True)
fi

# ========================
# Optional: enable DeepSpeed ZeRO stage-1
# Usage:
#   USE_DEEPSPEED=1 bash scripts/pretrain/nepa_b.sh
#   DEEPSPEED_CONFIG=configs/deepspeed/zero_stage1.json bash scripts/pretrain/nepa_b.sh
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
if [[ -z "${DEEPSPEED_CONFIG}" && "${USE_DEEPSPEED:-0}" == "1" ]]; then
    case "${MIXED_PRECISION}" in
        bf16) DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_bf16.json" ;;
        fp16) DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_fp16.json" ;;
        *) DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1.json" ;;
    esac
fi
DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

# ========================
export WANDB_PROJECT=$WANDB_PROJECT

# ========================
torchrun \
    --nnodes=$WORLD_SIZE  \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node $NGPU run_nepa.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --config_name $CONFIG_NAME \
    --image_processor_name  $CONFIG_NAME \
    --dataset_name $DATASET_PATH \
    --load_from_disk True \
    --dataloader_drop_last True \
    \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.025 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --optim adamw_torch \
    \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 50000 \
    \
    --seed 1337 \
    \
    "${GC_ARGS[@]}" \
    "${PRECISION_ARGS[@]}" \
    \
    "${DEEPSPEED_ARGS[@]}" \
    \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory False \
    \
    --report_to wandb \
    --run_name $EXPERIMENT_NAME
