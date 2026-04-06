import os

import torch

from src.data_processing.paths import PROCESSED_DIR


def with_split(payload, split_idx=0):
    out = dict(payload)
    for k in ("train_mask", "val_mask", "test_mask"):
        m = out[k]
        if isinstance(m, torch.Tensor) and m.dim() == 2:
            if split_idx < 0 or split_idx >= m.size(1):
                raise IndexError(
                    f"{k}: split_idx={split_idx} invalid for tensor with {m.size(1)} splits"
                )
            # one column -> [N] for train/eval
            out[k] = m[:, split_idx].contiguous()
    return out


def save_processed(name, payload):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, f"{name}.pt")
    torch.save(payload, path)
    return path


def load_processed(name):
    path = os.path.join(PROCESSED_DIR, f"{name}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    # PyTorch 2.6+ defaults weights_only=True; our payloads are trusted dicts of tensors.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
