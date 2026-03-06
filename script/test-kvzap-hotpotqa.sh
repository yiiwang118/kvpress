#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"
export KVZAP_TEST_MODEL="$HOME/src/ckpt/Qwen3-8B"
export HOTPOTQA_DEV_PATH="$HOME/src/data/hotpotQA/hotpot_dev_fullwiki_v1.json"

export PYTHONPATH="."
python tests/test_kvzap_hotpotqa.py
