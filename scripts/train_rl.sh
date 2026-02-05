#!/usr/bin/env bash
# train_rl.sh — simple, cloud-friendly trainer launcher
# Usage:
#   bash train_rl.sh \
#     --env-file /path/to/.env \
#     --data-file /path/to/fitness_scenarios.jsonl \
#     [--project-dir /path/to/fitness-reasoning-rl-agent] \
#     [--log-dir /path/to/logs]
#
# Required env (via --env-file or pre-exported):
#   OPENAI_API_KEY, PINECONE_API_KEY
# Optional:
#   WANDB_API_KEY

set -euo pipefail

# --- defaults ---
PROJECT_DIR="$(pwd)"
ENV_FILE=""
DATA_FILE=""
LOG_DIR="${PROJECT_DIR}/logs"

# --- args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --data-file) DATA_FILE="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

cd "$PROJECT_DIR"

# --- load env file if provided ---
if [[ -n "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file not found: ${ENV_FILE}" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# --- check required env ---
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
: "${PINECONE_API_KEY:?PINECONE_API_KEY is required}"

# --- ensure logs dir ---
mkdir -p "${LOG_DIR}"

# --- ensure data file is in expected path ---
# train.py expects "fitness_scenarios.jsonl" in PROJECT_DIR
TARGET_DATA="${PROJECT_DIR}/fitness_scenarios.jsonl"
if [[ -n "${DATA_FILE}" ]]; then
  if [[ ! -f "${DATA_FILE}" ]]; then
    echo "DATA_FILE not found: ${DATA_FILE}" >&2
    exit 1
  fi
  # create/refresh symlink or copy to expected filename
  if [[ -L "${TARGET_DATA}" || -f "${TARGET_DATA}" ]]; then
    rm -f "${TARGET_DATA}"
  fi
  ln -s "${DATA_FILE}" "${TARGET_DATA}"
fi

if [[ ! -f "${TARGET_DATA}" ]]; then
  echo "Missing required dataset file at ${TARGET_DATA}."
  echo "Provide it with: --data-file /path/to/fitness_scenarios.jsonl"
  exit 1
fi

# --- optional GPU check ---
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected NVIDIA GPU:"
  nvidia-smi || true
else
  echo "No nvidia-smi found. Proceeding without visible NVIDIA GPUs."
fi

# --- ensure uv is available; use uv for env + deps ---
ACTIVATE_CMD=""

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; attempting to install..."
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "curl/wget not found; installing curl via apt-get..."
      if command -v sudo >/dev/null 2>&1; then
        sudo -n apt-get update -y || sudo apt-get update -y || true
        sudo -n apt-get install -y curl ca-certificates || sudo apt-get install -y curl ca-certificates || true
      else
        apt-get update -y || true
        apt-get install -y curl ca-certificates || true
      fi
    fi
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh || true
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

if command -v uv >/dev/null 2>&1; then
  echo "Using uv to manage environment..."
  uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ACTIVATE_CMD="source .venv/bin/activate"

  if [[ -f "uv.lock" ]]; then
    uv sync --frozen
  else
    uv pip install -U pip
    uv pip install -e .
    uv pip install \
      jsonschema>=4.25.1 pydantic>=2.0.0 python-dotenv>=1.0.0 \
      "openpipe-art[backend,langgraph]>=0.4.11" \
      langchain-core>=0.3.0 langgraph>=0.2.0 langchain-openai>=0.2.0 \
      pinecone-client>=5.0.0 litellm>=1.0.0 \
      datasets>=2.14.0 pandas>=2.0.0 numpy>=1.24.0 \
      tenacity>=8.2.0 tqdm>=4.65.0 weave
  fi
else
  echo "Failed to install uv automatically. Falling back to python venv + pip..."
  PYTHON_bin="${PYTHON_BIN:-python3}"
  if ! command -v "${PYTHON_bin}" >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      echo "python3 not found. Attempting to install via apt-get..."
      if command -v sudo >/dev/null 2>&1; then
        sudo -n apt-get update -y || sudo apt-get update -y || true
        sudo -n apt-get install -y python3 python3-venv python3-pip || sudo apt-get install -y python3 python3-venv python3-pip || true
      else
        apt-get update -y || true
        apt-get install -y python3 python3-venv python3-pip || true
      fi
    else
      echo "python3 not found and apt-get unavailable. Please install Python 3.10+ manually." >&2
      exit 1
    fi
  fi
  "${PYTHON_bin}" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ACTIVATE_CMD="source .venv/bin/activate"
  pip install -U pip wheel setuptools
  pip install \
    jsonschema>=4.25.1 pydantic>=2.0.0 python-dotenv>=1.0.0 \
    "openpipe-art[backend,langgraph]>=0.4.11" \
    langchain-core>=0.3.0 langgraph>=0.2.0 langchain-openai>=0.2.0 \
    pinecone-client>=5.0.0 litellm>=1.0.0 \
    datasets>=2.14.0 pandas>=2.0.0 numpy>=1.24.0 \
    tenacity>=8.2.0 tqdm>=4.65.0 weave
fi

# --- print summary of keys presence (not values) ---
echo "Env summary:"
[[ -n "${OPENAI_API_KEY:-}" ]] && echo "  OPENAI_API_KEY: set"
[[ -n "${PINECONE_API_KEY:-}" ]] && echo "  PINECONE_API_KEY: set"
[[ -n "${WANDB_API_KEY:-}" ]] && echo "  WANDB_API_KEY: set" || echo "  WANDB_API_KEY: not set (logging to Weave disabled)"

# --- mkdir for ART local backend cache ---
mkdir -p "${PROJECT_DIR}/.art"

# --- run training ---
ts="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${ts}.log"
echo "Starting training... logs -> ${LOG_FILE}"
echo "To re-enter venv later: ${ACTIVATE_CMD}"

# Use python directly; entrypoint is async-compatible in train.py
python scripts/rag_fitnessrl_art_v2.py 2>&1 | tee -a "${LOG_FILE}"