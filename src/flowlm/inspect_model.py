"""Print the model config and run a greedy sanity generation.

Usage: uv run python -m flowlm.inspect_model [--device cpu] [--prompt "..."]
"""

import argparse
import json

import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from flowlm.model import DEFAULT_MODEL, config_summary, greedy_generate, load_text_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()

    supported = "qwen3_5" in CONFIG_MAPPING
    print(f"transformers {transformers.__version__}; qwen3_5 supported: {supported}")
    model, tokenizer = load_text_model(args.model, device=args.device)
    print(f"model class: {type(model).__name__}")
    print(json.dumps(config_summary(model.config), indent=2))

    n_params = sum(p.numel() for p in model.parameters())
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print(f"parameters: {n_params:,}; lm_head tied to embed_tokens: {tied}")

    text = greedy_generate(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"prompt: {args.prompt!r}\ncompletion: {text!r}")


if __name__ == "__main__":
    main()
