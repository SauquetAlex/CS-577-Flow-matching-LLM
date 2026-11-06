# CS-577-Flow-matching-LLM

Exploring flow matching on top of a modified [Qwen3.5](https://huggingface.co/Qwen) language model. This is a project for CS-577 at Purdue University.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked
```

## Loading the model

We use [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) as a **text-only** causal LM. The Hub checkpoint is multimodal, so `flowlm` builds only the text model from its `text_config`; the vision tower and MTP head are never instantiated (their weights are skipped at load time).

The first load downloads ~1.75 GB from the Hugging Face Hub into `~/.cache/huggingface/`. No account is needed; setting `HF_TOKEN` only raises the download rate limit.

Quick check (prints the config and greedy-generates 30 tokens):

```bash
uv run python -m flowlm.inspect_model --device cpu
# options: --device <torch device, e.g. cpu / cuda / mps>  --prompt "..."  --max-new-tokens 30
```

From Python:

```python
import torch
from flowlm.model import load_text_model, greedy_generate

model, tokenizer = load_text_model(device="cpu", dtype=torch.float32)
print(greedy_generate(model, tokenizer, "The capital of France is", max_new_tokens=30))
```

`load_text_model` returns a `transformers` `Qwen3_5ForCausalLM` in eval mode. The input embedding and the LM head share one weight matrix, so `model.lm_head.weight is model.model.embed_tokens.weight`. Read model dimensions from `model.config` (e.g. `hidden_size`, `layer_types`) instead of hardcoding them.

`transformers` will warn that the Gated DeltaNet layers are using its pure-PyTorch fallback because `flash-linear-attention` / `causal_conv1d` are not installed. The output is correct, just slower; we deliberately avoid hardware-specific kernels.

## Running tests

```bash
uv run pytest
```

The model tests download the checkpoint on first run and take about 30 s on CPU.

## Before pushing

Use these checks locally for formatting and linting before pushing. Note that GitHub CI runs the same checks on every pull request, and PRs can’t be merged until they pass.

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
