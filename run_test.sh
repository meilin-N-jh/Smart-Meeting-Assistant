#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh || true
conda activate cs6493 || conda activate qwen2.5

cd /home/jiahuning2/LLM_Ability_Test/CS6493
pip install httpx loguru python-dateutil
python3 scripts/evaluate.py
