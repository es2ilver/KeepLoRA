#!/bin/bash

#SBATCH --job-name=KeepLoRA_train_mixed2_epoch1_repo_bs8
#SBATCH --output=/shared/jeongeun/logs/%x-%j.out
#SBATCH --error=/shared/jeongeun/logs/%x-%j.err
#SBATCH --partition=batch
#SBATCH --nodelist=vgi2
#SBATCH --gres=gpu:rtx_4090:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=10G
#SBATCH --time=0-08:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Master port 설정 (여러 job 동시 실행 시 포트 충돌 방지)
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 10000))}"
else
  export MASTER_PORT="${MASTER_PORT:-29500}"
fi
echo "[MASTER_PORT=${MASTER_PORT}]"

# ---------------------------------------------------------------------------
# 경로 및 기본 설정 (필요하면 환경변수로 덮어쓰기)
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORK_DIR="/data/jeongeun"
CKPT_DIR="${WORK_DIR}/checkpoints"

IMAGE_FOLDER="${IMAGE_FOLDER:-${WORK_DIR}/datasets/MMMC_train}"
DATA_PATH="${DATA_PATH:-${WORK_DIR}/datasets/vqa_ctx_train_8k_llava_mixed_levels_0_1_2_3_4.json}"

SAVE_DIR="${SAVE_DIR:-${WORK_DIR}/checkpoints/}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAVE_DIR}/llava-v1.5-7b-lora_mixed_1epoch_keeplora_repo}"

# KeepLoRA 관련 설정
KEEP_DIR="${KEEP_DIR:-${SAVE_DIR}/keeplora_single}"
LORA_R="${LORA_R:-64}"
DATA_RATIO="${DATA_RATIO:-0.2}"
EPS_W="${EPS_W:-0.6}"   # epsilon_w (weight subspace)

mkdir -p "${KEEP_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "[ERROR] 데이터셋이 없습니다: ${DATA_PATH}"
  exit 1
fi
echo "[INFO] 학습 데이터: ${DATA_PATH}"
echo "[INFO] 학습 출력: ${OUTPUT_DIR}"
echo "[INFO] KeepLoRA 작업 디렉토리: ${KEEP_DIR}"

cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 0) Task0: 프리트레인 LLaVA 가중치에서 weight subspace 추출 (epsilon_w = EPS_W)
#    출력: ${W_SUBSPACE_DIR}/gradients_principle_subspace.pt
# ---------------------------------------------------------------------------
W_SUBSPACE_DIR="${KEEP_DIR}/Task0_llava_lora_gradients"
MODEL_CONFIG_TASK0="${KEEP_DIR}/model_task0.json"

cat > "${MODEL_CONFIG_TASK0}" <<EOF
{
  "model_path": "${CKPT_DIR}/llava-v1.5-7b"
}
EOF

echo "[STEP 0] Weight subspace 추출 (epsilon_w=${EPS_W})"
python -m llava.train.extract_weights \
  --model-config "${MODEL_CONFIG_TASK0}" \
  --output-dir "${W_SUBSPACE_DIR}" \
  --energy-threshold "${EPS_W}" \
  --target-modules "down_proj,q_proj,v_proj,o_proj,gate_proj,up_proj,k_proj"

# ---------------------------------------------------------------------------
# 1) 내 데이터셋에 대해 gradient subspace 추출 (fixed_rank, rank=LORA_R)
#    - data_ratio = DATA_RATIO (기본 0.2)
#    - 출력:
#        ${GRAD_SUBSPACE_DIR}/gradients_merged.pt
# ---------------------------------------------------------------------------
GRAD_SUBSPACE_DIR="${KEEP_DIR}/mixed_llava_lora_gradients"
MODEL_CONFIG_PRE="${KEEP_DIR}/model_pre.json"
DATA_CONFIG="${KEEP_DIR}/data_mixed.json"

cat > "${MODEL_CONFIG_PRE}" <<EOF
{
  "gpu_num": 1,
  "stage": "LoRA-single",
  "model_path": "${CKPT_DIR}/llava-v1.5-7b",
  "space_path": "${W_SUBSPACE_DIR}",
  "rank": ${LORA_R},
  "output_dir": "${GRAD_SUBSPACE_DIR}"
}
EOF

cat > "${DATA_CONFIG}" <<EOF
{
  "train_path": "${DATA_PATH}",
  "test_path": "${DATA_PATH}",
  "image_folder": "${IMAGE_FOLDER}"
}
EOF

echo "[STEP 1-1] Gradient 추출 (data_ratio=${DATA_RATIO}, rank=${LORA_R})"
python -m llava.train.extract_gradients \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG_PRE}" \
  --output-dir "${GRAD_SUBSPACE_DIR}" \
  --data-ratio "${DATA_RATIO}" \
  --batch-size 1 \
  --num-workers 4 \
  --num-chunks 1 \
  --chunk-idx 0

echo "[STEP 1-2] Gradient chunk 병합 + SVD (fixed_rank)"
python -m llava.train.extract_gradients \
  --data-config "${DATA_CONFIG}" \
  --model-config "${MODEL_CONFIG_PRE}" \
  --output-dir "${GRAD_SUBSPACE_DIR}" \
  --num-chunks 1 \
  --merge-only \
  --svd-mode fixed_rank

GRAD_INIT_PATH="${GRAD_SUBSPACE_DIR}/gradients_merged.pt"

if [[ ! -f "${GRAD_INIT_PATH}" ]]; then
  echo "[ERROR] gradients_merged.pt 가 생성되지 않았습니다: ${GRAD_INIT_PATH}"
  exit 1
fi
echo "[INFO] LoRA 초기화용 gradient subspace: ${GRAD_INIT_PATH}"

# ---------------------------------------------------------------------------
# 2) LLaVA + KeepLoRA 단일 task 학습
# ---------------------------------------------------------------------------
common_args=(
  --lora_enable True --lora_r "${LORA_R}" --lora_alpha 256 --mm_projector_lr 2e-5
  --freeze_lora_A True
  --init_lora_from_gradients "${GRAD_INIT_PATH}"
  --deepspeed ./scripts/zero3.json
  --model_name_or_path "${CKPT_DIR}/llava-v1.5-7b"
  --version v1
  --image_folder "${IMAGE_FOLDER}"
  --vision_tower openai/clip-vit-large-patch14-336
  --mm_projector_type mlp2x_gelu
  --mm_vision_select_layer -2
  --mm_use_im_start_end False
  --mm_use_im_patch_token False
  --image_aspect_ratio pad
  --group_by_modality_length True
  --bf16 True
  --num_train_epochs 1
  --per_device_train_batch_size 8
  --per_device_eval_batch_size 4
  --gradient_accumulation_steps 1
  --evaluation_strategy "no"
  --save_strategy "steps"
  --save_steps 50000
  --save_total_limit 1
  --learning_rate 2e-4
  --weight_decay 0.
  --warmup_ratio 0.03
  --lr_scheduler_type "cosine"
  --logging_steps 1
  --tf32 True
  --model_max_length 2048
  --gradient_checkpointing True
  --dataloader_num_workers 4
  --lazy_preprocess True
)

echo "[TRAIN] LLaVA + KeepLoRA (single task)"
deepspeed llava/train/train_mem.py \
  "${common_args[@]}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}"

echo "[DONE] 학습 완료 -> ${OUTPUT_DIR}"

