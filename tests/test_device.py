import os
import subprocess
import sys

import pytest
import torch

from flowlm.device import DEVICE_ENV_VAR, resolve_device


def test_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    assert resolve_device() == torch.device("cpu")


def test_reads_env_var(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "meta")
    assert resolve_device() == torch.device("meta")


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "meta")
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_invalid_device_raises(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, "not-a-device")
    with pytest.raises(RuntimeError):
        resolve_device()


def _device_seen_in_fresh_process(cwd, env_overrides):
    """Import flowlm in a new interpreter (so .env is loaded fresh) from ``cwd``."""
    env = {k: v for k, v in os.environ.items() if k != DEVICE_ENV_VAR}
    env.update(env_overrides)
    code = "import flowlm.device as d; print(d.resolve_device())"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_dotenv_file_sets_default(tmp_path):
    (tmp_path / ".env").write_text(f"{DEVICE_ENV_VAR}=meta\n")
    assert _device_seen_in_fresh_process(tmp_path, {}) == "meta"


def test_dotenv_found_from_subdirectory(tmp_path):
    (tmp_path / ".env").write_text(f"{DEVICE_ENV_VAR}=meta\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert _device_seen_in_fresh_process(sub, {}) == "meta"


def test_real_env_var_beats_dotenv(tmp_path):
    (tmp_path / ".env").write_text(f"{DEVICE_ENV_VAR}=meta\n")
    assert _device_seen_in_fresh_process(tmp_path, {DEVICE_ENV_VAR: "cpu"}) == "cpu"


def test_no_dotenv_falls_back_to_cpu(tmp_path):
    assert _device_seen_in_fresh_process(tmp_path, {}) == "cpu"
