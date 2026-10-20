import logging

import torch

log = logging.getLogger("playground")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
log.info("device: %s", DEVICE)
