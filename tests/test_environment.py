def test_environment_imports():
    import kvpress
    from kvpress import ExpectedAttentionPress, KVPressTextGenerationPipeline

    import accelerate
    import cachetools
    import datasets
    import fire
    import google.protobuf
    import nltk
    import numpy
    import pandas
    import requests
    import rouge
    import scipy
    import sentencepiece
    import torch
    import tqdm
    import transformers
    import bert_score

    assert kvpress is not None
    assert ExpectedAttentionPress is not None
    assert KVPressTextGenerationPipeline is not None
    assert torch.__version__
    assert transformers.__version__


def test_random_press_compresses_cache():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpress import RandomPress

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    press = RandomPress(compression_ratio=0.5, seed=0)
    with press(model):
        outputs = model(input_ids, use_cache=True)

    cache = outputs.past_key_values
    keys = cache.layers[0].keys
    values = cache.layers[0].values

    assert keys.shape[2] == 4
    assert values.shape[2] == 4


def test_local_llama_pipeline_generation_with_press():
    import os

    import pytest
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    import kvpress
    from kvpress import RandomPress

    model_path = "/data/images/llms/Meta-Llama-3.1-8B-Instruct"
    if not os.path.isdir(model_path):
        pytest.skip(f"Local model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if use_cuda else None,
    )

    pipe_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "model_kwargs": {"attn_implementation": "eager"},
    }
    if not use_cuda:
        pipe_kwargs["device"] = -1

    pipe = pipeline("kv-press-text-generation", **pipe_kwargs)

    press = RandomPress(compression_ratio=0.1)
    result = pipe("Hello", question="Say hi.", press=press, max_new_tokens=4, do_sample=False)

    assert "answer" in result
    assert isinstance(result["answer"], str)
