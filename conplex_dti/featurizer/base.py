from __future__ import annotations

import typing as T

from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from ..utils import get_logger

logg = get_logger()


def sanitize_string(s):
    return s.replace("/", "|")


def get_featurizer(featurizer_string, *args, **kwargs):
    from .. import featurizer as featurizers

    featurizer_string_list = featurizer_string.split(",")
    if len(featurizer_string_list) > 1:
        featurizer_list = [
            getattr(featurizers, i.strip()) for i in featurizer_string_list
        ]
        return featurizers.ConcatFeaturizer(featurizer_list, *args, **kwargs)
    else:
        return getattr(featurizers, featurizer_string_list[0])(*args, **kwargs)


###################
# Base Featurizer #
###################


class Featurizer:
    def __init__(self, name: str, shape: int, save_dir: Path = Path().absolute()):
        self._name = name
        self._shape = shape
        self._save_dir = save_dir
        self._save_path = save_dir / Path(f"{self._name}_features.h5")

        self._preloaded = False
        self._device = torch.device("cpu")
        self._cuda_registry = {}
        self._on_cuda = False
        self._features = {}

    def __call__(self, seq: str) -> torch.Tensor:
        if seq not in self.features:
            self._features[seq] = self.transform(seq)

        return self._features[seq]

    def _register_cuda(self, k: str, v, f=None):
        """
        Register an object as capable of being moved to a CUDA device
        """
        self._cuda_registry[k] = (v, f)

    def _transform(self, seq: str) -> torch.Tensor:
        raise NotImplementedError

    def _update_device(self, device: torch.device):
        self._device = device
        for k, (v, f) in self._cuda_registry.items():
            if f is None:
                try:
                    self._cuda_registry[k] = (v.to(self._device), None)
                except RuntimeError as e:
                    logg.error(e)
                    logg.debug(device)
                    logg.debug(type(self._device))
                    logg.debug(self._device)
            else:
                self._cuda_registry[k] = (f(v, self._device), f)
        for k, v in self._features.items():
            self._features[k] = v.to(device)

    @lru_cache(maxsize=5000)
    def transform(self, seq: str) -> torch.Tensor:
        with torch.set_grad_enabled(False):
            feats = self._transform(seq)
            if self._on_cuda:
                feats = feats.to(self.device)
            return feats

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def path(self) -> Path:
        return self._save_path

    @property
    def features(self) -> dict:
        return self._features

    @property
    def on_cuda(self) -> bool:
        return self._on_cuda

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> Featurizer:
        self._update_device(device)
        self._on_cuda = device.type == "cuda"
        return self

    def cuda(self, device: torch.device) -> Featurizer:
        """
        Perform model computations on CUDA, move saved embeddings to CUDA device
        """
        self._update_device(device)
        self._on_cuda = True
        return self

    def cpu(self) -> Featurizer:
        """
        Perform model computations on CPU, move saved embeddings to CPU
        """
        self._update_device(torch.device("cpu"))
        self._on_cuda = False
        return self

    def write_to_disk(self, seq_list: T.List[str], verbose: bool = True) -> None:
        logg.info(f"Writing {self.name} features to {self.path}")
        with h5py.File(self._save_path, "a") as h5fi:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                seq_h5 = sanitize_string(seq)
                if seq_h5 in h5fi:
                    logg.warning(f"{seq} already in h5file")
                feats = self.transform(seq)
                dset = h5fi.require_dataset(seq_h5, feats.shape, np.float32)
                dset[:] = feats.cpu().numpy()

    def preload(
        self,
        seq_list: T.List[str],
        verbose: bool = True,
        write_first: bool = True,
    ) -> None:
        logg.info(f"Preloading {self.name} features from {self.path}")

        if write_first and not self._save_path.exists():
            self.write_to_disk(seq_list, verbose=verbose)

        if self._save_path.exists():
            with h5py.File(self._save_path, "r") as h5fi:
                for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                    if seq in h5fi:
                        seq_h5 = sanitize_string(seq)
                        feats = torch.from_numpy(h5fi[seq_h5][:])
                    else:
                        feats = self.transform(seq)

                    if self._on_cuda:
                        feats = feats.to(self.device)

                    self._features[seq] = feats

        else:
            for seq in tqdm(seq_list, disable=not verbose, desc=self.name):
                feats = self.transform(seq)

                if self._on_cuda:
                    feats = feats.to(self.device)

                self._features[seq] = feats

        # seqs_sanitized = [sanitize_string(s) for s in seq_list]
        # feat_dict = load_hdf5_parallel(self._save_path, seqs_sanitized,n_jobs=32)
        # self._features.update(feat_dict)

        self._update_device(self.device)
        self._preloaded = True


class ConcatFeaturizer(Featurizer):
    def __init__(
        self,
        featurizer_list: T.Iterable[Featurizer],
        save_dir: Path = Path().absolute(),
    ):
        super().__init__("ConcatFeaturizer", None, save_dir)

        self._featurizer_list = featurizer_list
        self._featurizer_names = []
        for f in self._featurizer_list:
            featurizer = f(save_dir=save_dir)
            self._featurizer_names.append(featurizer._name)
            self._register_cuda(featurizer._name, featurizer)

        self._shape = sum(
            [self._cuda_registry[f_name][0].shape for f_name in self._featurizer_names]
        )

    def _transform(self, seq: str) -> torch.Tensor:
        feats = []
        for f_name in self._featurizer_names:
            featurizer = self._cuda_registry[f_name][0]
            feats.append(featurizer(seq))
        return torch.concat(feats)

    def write_to_disk(self, seq_list: T.List[str]) -> None:
        for f_name in self._featurizer_names:
            featurizer = self._cuda_registry[f_name][0]
            if not featurizer.path.exists():
                featurizer.write_to_disk(seq_list)

    def preload(self, seq_list: T.List[str], write_first: bool = True) -> None:
        for f_name in self._featurizer_names:
            featurizer = self._cuda_registry[f_name][0]
            featurizer.preload(seq_list, write_first=write_first)


###################
# Null and Random #
###################


class NullFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 1024,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__(f"Null{shape}", shape, save_dir)

    def _transform(self, seq: str) -> torch.Tensor:
        return torch.zeros(self.shape)


class RandomFeaturizer(Featurizer):
    def __init__(
        self,
        shape: int = 1024,
        save_dir: Path = Path().absolute(),
    ):
        super().__init__(f"Random{shape}", shape, save_dir)

    def _transform(self, seq: str) -> torch.Tensor:
        return torch.rand(self.shape)
