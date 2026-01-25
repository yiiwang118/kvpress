# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import KVzapPress, ThresholdPress


def _load_document() -> str:
    doc_path = os.getenv("KVZAP_DOC_PATH")
    if doc_path is None:
        raise RuntimeError("KVZAP_DOC_PATH must be set to run this script.")
    doc_path = Path(doc_path)
    return doc_path.read_text(encoding="utf-8")


def run_kvzap_document_compression():
    model_name = os.getenv("KVZAP_TEST_MODEL")
    if not model_name:
        raise RuntimeError("KVZAP_TEST_MODEL must be set to run this script.")

    press = ThresholdPress(KVzapPress(model_type="mlp"), threshold=-4)
    press.decoding = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)
    context = _load_document()
    question = "\nWhat is this document about in 2 sentences?"

    messages = [{"role": "user", "content": context + question}]
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    tokens = tokens.to(model.device)

    with press(model):
        output_tokens = model.generate(tokens, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_new_tokens=256)
    answer = tokenizer.decode(output_tokens[0, tokens.shape[1] :], skip_special_tokens=True)

    assert isinstance(answer, str)
    assert 0.0 <= press.compression_ratio <= 1.0


if __name__ == "__main__":
    run_kvzap_document_compression()
