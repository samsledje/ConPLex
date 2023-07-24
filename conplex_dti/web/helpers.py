import io

import torch

from . import models


def convert_tensor_to_column_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.to(models.BINARY_TORCH_TYPE).detach().cpu().numpy().tobytes()


def convert_column_bytes_to_tensor(column_bytes: bytes) -> torch.Tensor:
    return torch.frombuffer(column_bytes, dtype=models.BINARY_TORCH_TYPE)
