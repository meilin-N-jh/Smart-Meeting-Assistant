#!/bin/bash
# Smart Meeting Assistant - Startup Script

# This script helps start the Smart Meeting Assistant
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_DIR="/home/jiahuning2/LLM_Ability_Test/CS6493"
APP_ENV="${APP_ENV:-cs6493}"
MODEL_ENV="${MODEL_ENV:-qwen2.5}"

echo -e "${GREEN}Smart Meeting Assistant${NC}"
echo "=============================="

# Enter project dir first
cd "${PROJECT_DIR}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
fi

# Load env values for checks
set -a
source ./.env
set +a

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8400/v1}"
VLLM_MODELS_URL="${VLLM_BASE_URL%/}/models"

# Ensure localhost traffic never goes through proxy.
NO_PROXY_LOCAL="127.0.0.1,localhost,::1"
if [ -n "${NO_PROXY:-}" ]; then
    export NO_PROXY="${NO_PROXY_LOCAL},${NO_PROXY}"
else
    export NO_PROXY="${NO_PROXY_LOCAL}"
fi
export no_proxy="${NO_PROXY}"

# Check conda env
echo -e "\n${YELLOW}Checking conda environment...${NC}"
if ! command -v conda > /dev/null 2>&1; then
    echo -e "${RED}Conda not found!${NC}"
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fx "${APP_ENV}" > /dev/null; then
    PYTHON_VERSION=$(conda run -n "${APP_ENV}" python -V 2>&1)
    echo -e "${GREEN}App env: ${APP_ENV} (${PYTHON_VERSION})${NC}"
else
    echo -e "${RED}Conda env '${APP_ENV}' not found!${NC}"
    exit 1
fi

# Check if vLLM is running
echo -e "\n${YELLOW}Checking vLLM status at ${VLLM_MODELS_URL} ...${NC}"
if curl --noproxy '*' --max-time 5 -s "${VLLM_MODELS_URL}" > /dev/null 2>&1; then
    echo -e "${GREEN}vLLM is running!${NC}"
else
    echo -e "${RED}vLLM is NOT running!${NC}"
    echo "Please start vLLM first:"
    echo "  conda run -n ${MODEL_ENV} bash /home/jiahuning2/LLM_Ability_Test/models/Qwen2.5-7B/start_vllm_fp16.sh"
    exit 1
fi

# Start the application
echo -e "\n${YELLOW}Starting Smart Meeting Assistant...${NC}"
echo "Frontend: http://localhost:${PORT:-6493}"
echo "API Docs: http://localhost:${PORT:-6493}/docs"
echo -e "\nPress Ctrl+C to stop\n"

# Start the server
env ALL_PROXY= all_proxy= conda run -n "${APP_ENV}" python -m backend.main
