#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh || true
conda activate cs6493

export LLM_PROFILE="${LLM_PROFILE:-7b-fp16}"
export PORT="${PORT:-6493}"

cd /home/jiahuning2/LLM_Ability_Test/CS6493
echo "Starting Smart Meeting Assistant backend"
echo "LLM profile: ${LLM_PROFILE}"
echo "Port: ${PORT}"

uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}" --reload
