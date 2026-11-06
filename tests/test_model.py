import pytest
import torch
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from flowlm.model import greedy_generate, load_text_config, load_text_model

DEVICE = "cpu"


def test_transformers_supports_qwen3_5():
    assert "qwen3_5" in CONFIG_MAPPING
    assert "qwen3_5_text" in CONFIG_MAPPING


@pytest.fixture(scope="module")
def model_and_tokenizer():
    return load_text_model(device=DEVICE)


def test_text_config():
    config = load_text_config()
    assert config.model_type == "qwen3_5_text"
    assert config.layer_types is not None
    assert len(config.layer_types) == config.num_hidden_layers


def test_no_vision_or_mtp_modules(model_and_tokenizer):
    model, _ = model_and_tokenizer
    names = [n for n, _ in model.named_modules()]
    assert not any("visual" in n or "mtp" in n for n in names)


def test_dimensions_match_config(model_and_tokenizer):
    model, _ = model_and_tokenizer
    cfg = model.config
    assert model.model.embed_tokens.weight.shape == (cfg.vocab_size, cfg.hidden_size)
    assert len(model.model.layers) == cfg.num_hidden_layers


def test_embeddings_tied(model_and_tokenizer):
    model, _ = model_and_tokenizer
    assert model.config.tie_word_embeddings
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_weights_loaded_not_random(model_and_tokenizer):
    # A loaded LM assigns high probability to an obvious continuation.
    model, tok = model_and_tokenizer
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    paris = tok(" Paris", add_special_tokens=False).input_ids[0]
    assert logits.argmax().item() == paris


def test_greedy_generate_is_deterministic(model_and_tokenizer):
    model, tok = model_and_tokenizer
    a = greedy_generate(model, tok, "def fibonacci(n):", max_new_tokens=10)
    b = greedy_generate(model, tok, "def fibonacci(n):", max_new_tokens=10)
    assert a == b and len(a) > 0
