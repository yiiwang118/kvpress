#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

pytest tests/test_environment.py
