"""Choosing the torch device.

There is no hardware detection: the device is whatever the caller passes, or
else the ``FLOWLM_DEVICE`` environment variable, or else ``"cpu"``.
"""

import os

import torch

DEVICE_ENV_VAR = "FLOWLM_DEVICE"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return ``device`` if given, else ``$FLOWLM_DEVICE``, else ``cpu``."""
    if device is None:
        device = os.environ.get(DEVICE_ENV_VAR) or "cpu"
    return torch.device(device)
