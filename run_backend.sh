#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh || true
conda activate cs6493

export LLM_PROFILE="${LLM_PROFILE:-7b-fp16}"
export PORT="${PORT:-6493}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
export ASR_DEVICE="${ASR_DEVICE:-cuda:3}"
export DIARIZATION_DEVICE="${DIARIZATION_DEVICE:-cuda:3}"

cd /home/jiahuning2/LLM_Ability_Test/CS6493
echo "Starting Smart Meeting Assistant backend"
echo "LLM profile: ${LLM_PROFILE}"
echo "Port: ${PORT}"
echo "ASR device: ${ASR_DEVICE}"
echo "Diarization device: ${DIARIZATION_DEVICE}"

uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}" --reload
