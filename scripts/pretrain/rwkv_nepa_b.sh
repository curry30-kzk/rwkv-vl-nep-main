#!/usr/bin/env bash

set -euo pipefail

: "${NCCL_DEBUG:=INFO}"
: "${TORCH_DISTRIBUTED_TIMEOUT:=1800}"
: "${WANDB_MODE:=offline}"
: "${MIXED_PRECISION:=bf16}"
: "${USE_DEEPSPEED:=0}"
: "${USE_GC:=0}"
: "${REPORT_TO:=none}"
: "${DDP_FIND_UNUSED_PARAMETERS:=True}"

export NCCL_DEBUG
export TORCH_DISTRIBUTED_TIMEOUT
export WANDB_MODE
export MIXED_PRECISION
export USE_DEEPSPEED
export USE_GC
export REPORT_TO
export DDP_FIND_UNUSED_PARAMETERS

: "${WORLD_SIZE:=1}"
: "${RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"

: "${NGPU:=}"

if [[ -z "${NGPU}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
        NGPU="${#_GPU_ARR[@]}"
    else
        NGPU=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
    fi
fi

if ! [[ "${NGPU}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] NGPU is not a valid integer: ${NGPU}"
    exit 1
fi

if (( NGPU < 1 )); then
    echo "[ERROR] No visible GPU found."
    exit 1
fi

: "${EXPERIMENT_NAME:=rwkv-nepa-base-patch14-224}"
: "${WANDB_PROJECT:=RwkvNepa-Pretrain}"

: "${CONFIG_NAME:=configs/pretrain/rwkv-nepa-base-patch14-224}"
: "${DATASET_PATH:=/public/home/ssjxkzk/rwkv-vl-nep-main/data/imagenet-1k-hf}"
: "${OUTPUT_DIR:=outputs/${EXPERIMENT_NAME}}"

: "${TOTAL_BATCH_SIZE:=4096}"
: "${PER_DEVICE_BATCH_SIZE:=256}"
: "${NUM_EPOCHS:=8}"
: "${MAX_STEPS:=-1}"

: "${BASE_LEARNING_RATE:=3e-4}"
: "${WEIGHT_DECAY:=0.05}"
: "${WARMUP_RATIO:=0.025}"
: "${LR_SCHEDULER_TYPE:=cosine}"
: "${OPTIM:=adamw_torch}"

: "${LOGGING_STEPS:=50}"
: "${SAVE_STRATEGY:=steps}"
: "${SAVE_STEPS:=1000}"
: "${DATALOADER_NUM_WORKERS:=20}"
: "${DATALOADER_PERSISTENT_WORKERS:=True}"
: "${DATALOADER_PIN_MEMORY:=False}"

# Strict resume support.
: "${RESUME_FROM_CHECKPOINT:=}"

# Optional uint8 npy/mmap preprocessing experiment. Defaults keep the original Arrow path unchanged.
: "${USE_PREPROCESSED_UINT8:=False}"
: "${PREPROCESSED_DATA_DIR:=}"

# Optional pretraining EMA controls exposed by run_nepa.py.
: "${USE_EMA:=True}"
: "${EMA_DECAY:=0.9999}"
: "${EMA_UPDATE_INTERVAL:=1}"

if ! [[ "${TOTAL_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TOTAL_BATCH_SIZE must be an integer, got: ${TOTAL_BATCH_SIZE}"
    exit 1
fi
if ! [[ "${PER_DEVICE_BATCH_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] PER_DEVICE_BATCH_SIZE must be an integer, got: ${PER_DEVICE_BATCH_SIZE}"
    exit 1
fi
if ! [[ "${WORLD_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] WORLD_SIZE must be an integer, got: ${WORLD_SIZE}"
    exit 1
fi
if ! [[ "${MAX_STEPS}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] MAX_STEPS must be an integer, got: ${MAX_STEPS}"
    exit 1
fi

if (( PER_DEVICE_BATCH_SIZE < 1 )); then
    echo "[ERROR] PER_DEVICE_BATCH_SIZE must be >= 1"
    exit 1
fi
if (( WORLD_SIZE < 1 )); then
    echo "[ERROR] WORLD_SIZE must be >= 1"
    exit 1
fi

DENOM=$(( PER_DEVICE_BATCH_SIZE * NGPU * WORLD_SIZE ))
if (( DENOM < 1 )); then
    echo "[ERROR] Invalid denominator for gradient accumulation: ${DENOM}"
    exit 1
fi

if (( TOTAL_BATCH_SIZE % DENOM != 0 )); then
    echo "[ERROR] TOTAL_BATCH_SIZE must be divisible by PER_DEVICE_BATCH_SIZE * NGPU * WORLD_SIZE"
    echo "        TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
    echo "        PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    echo "        NGPU=${NGPU}"
    echo "        WORLD_SIZE=${WORLD_SIZE}"
    echo "        denominator=${DENOM}"
    exit 1
fi

GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / DENOM ))

LEARNING_RATE=$(python - <<PY
base_lr = float("${BASE_LEARNING_RATE}")
total_bs = int("${TOTAL_BATCH_SIZE}")
print(base_lr * total_bs / 256)
PY
)

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
        echo "[ERROR] Unknown MIXED_PRECISION=${MIXED_PRECISION}. Use auto|bf16|fp16|fp32."
        exit 1
        ;;
esac

GC_ARGS=()
if [[ "${USE_GC}" == "1" ]]; then
    GC_ARGS=(--gradient_checkpointing True)
fi

DEEPSPEED_ARGS=()
if [[ "${USE_DEEPSPEED}" == "1" ]]; then
    if ! python - <<'PY' >/dev/null 2>&1
import deepspeed  # noqa: F401
PY
    then
        echo "[ERROR] USE_DEEPSPEED=1 but DeepSpeed is not installed in the current environment."
        exit 1
    fi

    : "${DEEPSPEED_CONFIG:=}"
    if [[ -z "${DEEPSPEED_CONFIG}" ]]; then
        case "${MIXED_PRECISION}" in
            bf16) DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_bf16.json" ;;
            fp16) DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1_fp16.json" ;;
            *) DEEPSPEED_CONFIG="configs/deepspeed/zero_stage1.json" ;;
        esac
    fi
    DEEPSPEED_ARGS=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

export WANDB_PROJECT="${WANDB_PROJECT}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-}"
TRAINABLE="${TRAINABLE:-}"
EMBED_LR="${EMBED_LR:-}"

EXTRA_MODEL_ARGS=()
if [[ -n "${MODEL_NAME_OR_PATH}" ]]; then
    EXTRA_MODEL_ARGS+=(--model_name_or_path "${MODEL_NAME_OR_PATH}")
fi
if [[ -n "${TRAINABLE}" ]]; then
    EXTRA_MODEL_ARGS+=(--trainable "${TRAINABLE}")
fi
if [[ -n "${EMBED_LR}" ]]; then
    EXTRA_MODEL_ARGS+=(--embed_lr "${EMBED_LR}")
fi

TRAINING_LENGTH_ARGS=()
if (( MAX_STEPS > 0 )); then
    TRAINING_LENGTH_ARGS+=(--max_steps "${MAX_STEPS}")
else
    TRAINING_LENGTH_ARGS+=(--num_train_epochs "${NUM_EPOCHS}")
fi

SAVE_ARGS=()
if [[ "${SAVE_STRATEGY}" == "no" ]]; then
    SAVE_ARGS+=(--save_strategy no)
elif [[ "${SAVE_STRATEGY}" == "epoch" ]]; then
    SAVE_ARGS+=(--save_strategy epoch)
else
    SAVE_ARGS+=(--save_strategy steps --save_steps "${SAVE_STEPS}")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

PREPROCESSED_ARGS=()
if [[ "${USE_PREPROCESSED_UINT8,,}" == "true" || "${USE_PREPROCESSED_UINT8}" == "1" ]]; then
    if [[ -z "${PREPROCESSED_DATA_DIR}" ]]; then
        echo "[ERROR] PREPROCESSED_DATA_DIR is required when USE_PREPROCESSED_UINT8=True."
        exit 1
    fi
    PREPROCESSED_ARGS+=(--use_preprocessed_uint8 True --preprocessed_data_dir "${PREPROCESSED_DATA_DIR}")
fi

EMA_ARGS=(
    --use_ema "${USE_EMA}"
    --ema_decay "${EMA_DECAY}"
    --ema_update_interval "${EMA_UPDATE_INTERVAL}"
)

echo "[INFO] EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] NGPU=${NGPU}, WORLD_SIZE=${WORLD_SIZE}, RANK=${RANK}"
echo "[INFO] TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}, PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}, GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "[INFO] MIXED_PRECISION=${MIXED_PRECISION}, USE_DEEPSPEED=${USE_DEEPSPEED}, USE_GC=${USE_GC}, REPORT_TO=${REPORT_TO}"
echo "[INFO] DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS}"
echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}, NUM_EPOCHS=${NUM_EPOCHS}"
echo "[INFO] LOGGING_STEPS=${LOGGING_STEPS}, SAVE_STRATEGY=${SAVE_STRATEGY}, SAVE_STEPS=${SAVE_STEPS}"
echo "[INFO] OPTIM=${OPTIM}"
echo "[INFO] DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}, DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS}, DATALOADER_PIN_MEMORY=${DATALOADER_PIN_MEMORY}"
echo "[INFO] USE_EMA=${USE_EMA}, EMA_DECAY=${EMA_DECAY}, EMA_UPDATE_INTERVAL=${EMA_UPDATE_INTERVAL}"
if [[ -n "${MODEL_NAME_OR_PATH}" ]]; then
    echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
fi
if [[ -n "${TRAINABLE}" ]]; then
    echo "[INFO] TRAINABLE=${TRAINABLE}"
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "[INFO] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
fi
if [[ "${USE_DEEPSPEED}" == "1" ]]; then
    echo "[INFO] DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
fi
if [[ "${USE_PREPROCESSED_UINT8,,}" == "true" || "${USE_PREPROCESSED_UINT8}" == "1" ]]; then
    echo "[INFO] USE_PREPROCESSED_UINT8=${USE_PREPROCESSED_UINT8}"
    echo "[INFO] PREPROCESSED_DATA_DIR=${PREPROCESSED_DATA_DIR}"
fi

torchrun \
    --nnodes="${WORLD_SIZE}" \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node "${NGPU}" \
    run_nepa.py \
    --ddp_backend nccl \
    --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}" \
    --config_name "${CONFIG_NAME}" \
    --image_processor_name "${CONFIG_NAME}" \
    "${EXTRA_MODEL_ARGS[@]}" \
    --dataset_name "${DATASET_PATH}" \
    --load_from_disk True \
    "${PREPROCESSED_ARGS[@]}" \
    --dataloader_drop_last True \
    --do_train \
    --output_dir "${OUTPUT_DIR}" \
    --remove_unused_columns False \
    "${TRAINING_LENGTH_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --optim "${OPTIM}" \
    --logging_strategy steps \
    --logging_steps "${LOGGING_STEPS}" \
    "${SAVE_ARGS[@]}" \
    "${EMA_ARGS[@]}" \
    --seed 1337 \
    "${GC_ARGS[@]}" \
    "${PRECISION_ARGS[@]}" \
    "${DEEPSPEED_ARGS[@]}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}" \
    --dataloader_pin_memory "${DATALOADER_PIN_MEMORY}" \
    --report_to "${REPORT_TO}" \
    --run_name "${EXPERIMENT_NAME}"
