import io

import numpy as np
import torch

from . import models


def convert_array_to_column_bytes(array: np.ndarray) -> bytes:
    return array.astype(models.BINARY_NUMPY_TYPE, copy=False).tobytes()


def convert_column_bytes_to_array(column_bytes: bytes) -> np.ndarray:
    return np.frombuffer(column_bytes, dtype=models.BINARY_NUMPY_TYPE)


def convert_tensor_to_column_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.to(models.BINARY_TORCH_TYPE).detach().cpu().numpy().tobytes()


def convert_column_bytes_to_tensor(column_bytes: bytes) -> torch.Tensor:
    return torch.frombuffer(column_bytes, dtype=models.BINARY_TORCH_TYPE)
