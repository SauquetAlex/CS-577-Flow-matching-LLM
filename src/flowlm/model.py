"""Loading Qwen3.5 as a text-only causal LM.

The Hub checkpoint is a multimodal ``Qwen3_5ForConditionalGeneration`` (vision
tower + MTP head). We take only its ``text_config`` and build
``Qwen3_5ForCausalLM`` from it, so no vision or MTP modules are instantiated;
their checkpoint weights are skipped at load time.
"""

from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from flowlm.device import resolve_device

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"


def load_text_config(name: str = DEFAULT_MODEL) -> Qwen3_5TextConfig:
    """Return the text sub-config of a Qwen3.5 checkpoint."""
    config = AutoConfig.from_pretrained(name)
    text_config = getattr(config, "text_config", config)
    if not isinstance(text_config, Qwen3_5TextConfig):
        raise TypeError(f"expected Qwen3_5TextConfig, got {type(text_config).__name__}")
    return text_config


def load_text_model(
    name: str = DEFAULT_MODEL,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Qwen3_5ForCausalLM, PreTrainedTokenizerBase]:
    """Load the text-only causal LM and its tokenizer, in eval mode on ``device``.

    ``device=None`` means ``$FLOWLM_DEVICE`` if set, else ``cpu``.
    """
    config = load_text_config(name)
    model = Qwen3_5ForCausalLM.from_pretrained(name, config=config, dtype=dtype)
    model.to(resolve_device(device)).eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(f"no tokenizer found for {name}")
    return model, tokenizer


def config_summary(config: Qwen3_5TextConfig) -> dict[str, Any]:
    """The config fields we depend on downstream."""
    # Optional in the config's type hints, but always filled in by __post_init__.
    layer_types, rope = config.layer_types, config.rope_parameters
    if layer_types is None or rope is None:
        raise ValueError("config is missing layer_types or rope_parameters")
    return {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "layer_types": list(layer_types),
        "tie_word_embeddings": config.tie_word_embeddings,
        "vocab_size": config.vocab_size,
        "rope_parameters": dict(rope),
        "max_position_embeddings": config.max_position_embeddings,
    }


@torch.no_grad()
def greedy_generate(
    model: Qwen3_5ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 30,
) -> str:
    """Greedy-decode ``max_new_tokens`` tokens after ``prompt``; return the new text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # ty: transformers' GenerativePreTrainedModel protocol rejects every HF model
    # (its `_supports_logits_to_keep` signature disagrees with GenerationMixin's).
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)  # ty: ignore[invalid-argument-type]
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    assert isinstance(text, str)  # 1-D ids decode to a single string
    return text
