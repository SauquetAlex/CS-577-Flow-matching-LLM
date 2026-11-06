# CS-577-Flow-matching-LLM

Exploring flow matching on top of a modified [Qwen3.5](https://huggingface.co/Qwen) language model. This is a project for CS-577 at Purdue University.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked
```

## Before pushing

Use these checks locally for formatting and linting before pushing. Note that GitHub CI runs the same checks on every pull request, and PRs can’t be merged until they pass.

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
```
