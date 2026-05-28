# ========================
# Runs a short RWKV-NEPA speed comparison:
# 1) original Arrow load_from_disk path
# 2) preprocessed uint8 npy/mmap path
#
# Required for the uint8 run:
#   PREPROCESSED_DATA_DIR=/path/to/cache
#
# Useful overrides:
#   DATASET_PATH=data/imagenet-1k-hf
#   MAX_STEPS=20
#   BENCH_ROOT=outputs/bench_uint8_vs_arrow
# ========================

: "${DATASET_PATH:=data/imagenet-1k-hf}"
: "${PREPROCESSED_DATA_DIR:=}"
: "${MAX_STEPS:=20}"
: "${BENCH_ROOT:=outputs/bench_uint8_vs_arrow}"
: "${RUN_ID:=$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${PREPROCESSED_DATA_DIR}" ]]; then
    echo "PREPROCESSED_DATA_DIR is required."
    exit 1
fi

ARROW_OUTPUT_DIR="${BENCH_ROOT}/${RUN_ID}_arrow"
UINT8_OUTPUT_DIR="${BENCH_ROOT}/${RUN_ID}_uint8"

COMMON_ENV=(
    "DATASET_PATH=${DATASET_PATH}"
    "MAX_STEPS=${MAX_STEPS}"
    "NUM_EPOCHS=1"
    "USE_EMA=False"
    "USE_GC=0"
    "USE_DEEPSPEED=0"
    "LOGGING_STEPS=1"
    "SAVE_STEPS=100000"
)

echo "[1/2] Arrow baseline -> ${ARROW_OUTPUT_DIR}"
env \
    "${COMMON_ENV[@]}" \
    "EXPERIMENT_NAME=bench_arrow" \
    "OUTPUT_DIR=${ARROW_OUTPUT_DIR}" \
    "USE_PREPROCESSED_UINT8=False" \
    bash scripts/pretrain/rwkv_nepa_b.sh

echo "[2/2] uint8 npy/mmap -> ${UINT8_OUTPUT_DIR}"
env \
    "${COMMON_ENV[@]}" \
    "EXPERIMENT_NAME=bench_uint8" \
    "OUTPUT_DIR=${UINT8_OUTPUT_DIR}" \
    "USE_PREPROCESSED_UINT8=True" \
    "PREPROCESSED_DATA_DIR=${PREPROCESSED_DATA_DIR}" \
    bash scripts/pretrain/rwkv_nepa_b.sh

python compare_train_speed.py \
    --arrow_output_dir "${ARROW_OUTPUT_DIR}" \
    --uint8_output_dir "${UINT8_OUTPUT_DIR}"
