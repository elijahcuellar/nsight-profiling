from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(message: str):
    pushed = False
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        try:
            torch.cuda.nvtx.range_push(message)
            pushed = True
        except Exception:
            pushed = False

    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


def get_device() -> str:
    try:
        return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    except Exception:
        return "cpu"


def default_classes():
    return [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
