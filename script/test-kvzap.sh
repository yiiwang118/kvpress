#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"
export KVZAP_TEST_MODEL="$HOME/src/ckpt/Qwen3-8B"
export KVZAP_DOC_PATH="tests/data/kvzap_doc.txt"

export PYTHONPATH="."
python tests/test_kvzap.py
