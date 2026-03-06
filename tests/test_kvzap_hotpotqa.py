# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import re
import string
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import KVzapPress, ThresholdPress


def _load_hotpotqa_examples(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"HotpotQA file is empty: {path}")
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise RuntimeError("HotpotQA JSON must be a list of examples.")
        return data
    examples = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return [tok for tok in text.split() if tok]


def _build_passages(example: dict, sentences_per_passage: int) -> list[tuple[str, str]]:
    context = example.get("context", [])
    passages = []
    for title, sent_list in context:
        for i in range(0, len(sent_list), sentences_per_passage):
            chunk = " ".join(sent_list[i : i + sentences_per_passage])
            if chunk:
                passages.append((title, f"{title}: {chunk}"))
    return passages


def _retrieve_passages(question: str, passages: list[tuple[str, str]], top_k: int) -> list[tuple[str, str]]:
    if not passages:
        return []
    query_tokens = _tokenize(question)
    doc_tokens = [set(_tokenize(text)) for _, text in passages]
    df = Counter()
    for tokens in doc_tokens:
        df.update(tokens)
    num_docs = len(passages)
    scores = []
    for idx, tokens in enumerate(doc_tokens):
        score = 0.0
        for tok in query_tokens:
            if tok in tokens:
                score += math.log((num_docs + 1) / (df[tok] + 1)) + 1.0
        scores.append((score, idx))
    scores.sort(key=lambda item: item[0], reverse=True)
    selected = [passages[idx] for _, idx in scores[:top_k]]
    return selected


def _rerank_passages(question: str, passages: list[tuple[str, str]], top_k: int) -> list[tuple[str, str]]:
    query_tokens = set(_tokenize(question))
    scored = []
    for title, text in passages:
        passage_tokens = _tokenize(text)
        overlap = sum(1 for tok in passage_tokens if tok in query_tokens)
        title_overlap = sum(1 for tok in _tokenize(title) if tok in query_tokens)
        score = overlap / max(len(passage_tokens), 1) + 0.1 * title_overlap
        scored.append((score, (title, text)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def _build_context(example: dict, max_sentences: int, retrieval_top_k: int, rerank_top_k: int) -> str:
    passages = _build_passages(example, sentences_per_passage=2)
    if not passages:
        return ""
    question = example.get("question", "")
    retrieved = _retrieve_passages(question, passages, top_k=retrieval_top_k)
    reranked = _rerank_passages(question, retrieved, top_k=rerank_top_k)
    sentences = []
    for _, text in reranked:
        sentences.append(text)
        if len(sentences) >= max_sentences:
            break
    return "\n".join(sentences)


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _batched(iterable: Iterable[dict], n: int) -> Iterable[list[dict]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _generate_answers(model, tokenizer, prompts: list[str], press=None) -> list[str]:
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, padding=True)
    tokens = tokens.to(model.device)
    attention_mask = tokens.ne(tokenizer.pad_token_id)
    context_mgr = press(model) if press is not None else nullcontext()
    with context_mgr:
        output_tokens = model.generate(
            tokens,
            attention_mask=attention_mask,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            max_new_tokens=64,
        )
    outputs = []
    for i in range(tokens.shape[0]):
        decoded = tokenizer.decode(output_tokens[i, tokens.shape[1] :], skip_special_tokens=True)
        outputs.append(decoded.strip())
    return outputs


def run_kvzap_hotpotqa_comparison():
    model_name = os.getenv("KVZAP_TEST_MODEL")
    if not model_name:
        raise RuntimeError("KVZAP_TEST_MODEL must be set to run this script.")

    hotpot_path = os.getenv("HOTPOTQA_DEV_PATH")
    if not hotpot_path:
        raise RuntimeError("HOTPOTQA_DEV_PATH must be set to run this script.")

    max_examples = int(os.getenv("HOTPOTQA_NUM_EXAMPLES", "20"))
    max_sentences = int(os.getenv("HOTPOTQA_MAX_CONTEXT_SENTENCES", "30"))
    retrieval_top_k = int(os.getenv("HOTPOTQA_RETRIEVAL_TOP_K", "20"))
    rerank_top_k = int(os.getenv("HOTPOTQA_RERANK_TOP_K", "8"))
    batch_size = int(os.getenv("HOTPOTQA_BATCH_SIZE", "2"))

    data = _load_hotpotqa_examples(Path(hotpot_path))
    examples = data[:max_examples]
    if not examples:
        raise RuntimeError("No HotpotQA examples loaded.")

    press = ThresholdPress(KVzapPress(model_type="mlp"), threshold=-4)
    press.decoding = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)

    baseline_em = []
    baseline_f1 = []
    kvzap_em = []
    kvzap_f1 = []

    for batch in _batched(examples, batch_size):
        prompts = []
        answers = []
        for example in batch:
            context = _build_context(
                example,
                max_sentences=max_sentences,
                retrieval_top_k=retrieval_top_k,
                rerank_top_k=rerank_top_k,
            )
            question = example.get("question", "")
            answer = example.get("answer", "")
            prompts.append(f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
            answers.append(answer)

        baseline_outputs = _generate_answers(model, tokenizer, prompts, press=None)
        kvzap_outputs = _generate_answers(model, tokenizer, prompts, press=press)

        for pred, gold in zip(baseline_outputs, answers, strict=True):
            baseline_em.append(_exact_match(pred, gold))
            baseline_f1.append(_f1_score(pred, gold))

        for pred, gold in zip(kvzap_outputs, answers, strict=True):
            kvzap_em.append(_exact_match(pred, gold))
            kvzap_f1.append(_f1_score(pred, gold))

    baseline_em_avg = sum(baseline_em) / len(baseline_em)
    baseline_f1_avg = sum(baseline_f1) / len(baseline_f1)
    kvzap_em_avg = sum(kvzap_em) / len(kvzap_em)
    kvzap_f1_avg = sum(kvzap_f1) / len(kvzap_f1)

    print(
        "HotpotQA results (n="
        f"{len(examples)}): baseline EM={baseline_em_avg:.3f}, F1={baseline_f1_avg:.3f} | "
        f"kvzap EM={kvzap_em_avg:.3f}, F1={kvzap_f1_avg:.3f} | "
        f"compression_ratio={press.compression_ratio:.3f}"
    )

    assert 0.0 <= press.compression_ratio <= 1.0
    assert len(baseline_em) == len(kvzap_em) == len(examples)


if __name__ == "__main__":
    run_kvzap_hotpotqa_comparison()
