import io

import torch

from . import models


def convert_tensor_to_column_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.to(models.BINARY_TORCH_TYPE)
    stream = io.BytesIO()
    torch.save(tensor, stream)
    return stream.getvalue()


def convert_column_bytes_to_tensor(column_bytes: bytes) -> torch.Tensor:
    return torch.frombuffer(column_bytes, dtype=models.BINARY_TORCH_TYPE)
