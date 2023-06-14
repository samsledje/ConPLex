import typing as T
from types import SimpleNamespace

import os
import pickle as pk
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from numpy.random import choice
from sklearn.model_selection import KFold, train_test_split
from tdc.benchmark_group import dti_dg_group
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ..featurizer import Featurizer
from ..featurizer.protein import FOLDSEEK_MISSING_IDX
from ..utils import get_logger

logg = get_logger()


def get_task_dir(task_name: str, database_root: Path):
    """
    Get the path to data for each benchmark data set

    :param task_name: Name of benchmark
    :type task_name: str
    """

    database_root = Path(database_root).resolve()

    task_paths = {
        "biosnap": database_root / "BIOSNAP/full_data",
        "biosnap_prot": database_root / "BIOSNAP/unseen_protein",
        "biosnap_mol": database_root / "BIOSNAP/unseen_drug",
        "bindingdb": database_root / "BindingDB",
        "davis": database_root / "DAVIS",
        "dti_dg": database_root / "TDC",
        "dude": database_root / "DUDe",
        "halogenase": database_root / "EnzPred/halogenase_NaCl_binary",
        "bkace": database_root / "EnzPred/duf_binary",
        "gt": database_root / "EnzPred/gt_acceptors_achiral_binary",
        "esterase": database_root / "EnzPred/esterase_binary",
        "kinase": database_root / "EnzPred/davis_filtered",
        "phosphatase": database_root / "EnzPred/phosphatase_chiral_binary",
    }

    return Path(task_paths[task_name.lower()]).resolve()


def drug_target_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    If target embeddings are not all the same length, it will zero pad them
    This is to account for differences in length from FoldSeek embeddings

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    d_emb = [a[0] for a in args]
    t_emb = [a[1] for a in args]
    labs = [a[2] for a in args]

    drugs = torch.stack(d_emb, 0)
    targets = pad_sequence(t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)
    labels = torch.stack(labs, 0)

    return drugs, targets, labels


def contrastive_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    Specific collate function for contrastive dataloader

    :param args: Batch of training samples with anchor, positive, negative
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    anchor_emb = [a[0] for a in args]
    pos_emb = [a[1] for a in args]
    neg_emb = [a[2] for a in args]

    anchors = pad_sequence(
        anchor_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX
    )
    positives = torch.stack(pos_emb, 0)
    negatives = torch.stack(neg_emb, 0)

    return anchors, positives, negatives


def make_contrastive(
    df: pd.DataFrame,
    posneg_column: str,
    anchor_column: str,
    label_column: str,
    n_neg_per: int = 50,
):
    pos_df = df[df[label_column] == 1]
    neg_df = df[df[label_column] == 0]

    contrastive = []

    for _, r in pos_df.iterrows():
        for _ in range(n_neg_per):
            contrastive.append(
                (
                    r[anchor_column],
                    r[posneg_column],
                    choice(neg_df[posneg_column]),
                )
            )

    contrastive = pd.DataFrame(contrastive, columns=["Anchor", "Positive", "Negative"])
    return contrastive


class BinaryDataset(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        labels,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
    ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])
        target = self.target_featurizer(self.targets.iloc[i])
        label = torch.tensor(self.labels.iloc[i])

        return drug, target, label


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        anchors,
        positives,
        negatives,
        posneg_featurizer: Featurizer,
        anchor_featurizer: Featurizer,
    ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

        self.posneg_featurizer = posneg_featurizer
        self.anchor_featurizer = anchor_featurizer

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        anchorEmb = self.anchor_featurizer(self.anchors[i])
        positiveEmb = self.posneg_featurizer(self.positives[i])
        negativeEmb = self.posneg_featurizer(self.negatives[i])

        return anchorEmb, positiveEmb, negativeEmb


class DTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")

        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):
        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")
            return

        df_train = pd.read_csv(self._data_dir / self._train_path, **self._csv_kwargs)

        df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)

        df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        dataframes = [df_train, df_val, df_test]
        all_drugs = pd.concat([i[self._drug_column] for i in dataframes]).unique()
        all_targets = pd.concat([i[self._target_column] for i in dataframes]).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):
        self.df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )

        self.df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)

        self.df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)


class TDCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._seed = seed

        self._drug_column = "Drug"
        self._target_column = "Target"
        self._label_column = "Y"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):
        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")

        train_val, test = (
            dg_benchmark["train_val"],
            dg_benchmark["test"],
        )

        all_drugs = pd.concat([train_val, test])[self._drug_column].unique()
        all_targets = pd.concat([train_val, test])[self._target_column].unique()

        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")
            return

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

    def setup(self, stage: T.Optional[str] = None):
        dg_group = dti_dg_group(path=self._data_dir)
        dg_benchmark = dg_group.get("bindingdb_patent")
        dg_name = dg_benchmark["name"]

        self.df_train, self.df_val = dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="default", seed=self._seed
        )
        self.df_test = dg_benchmark["test"]

        self._dataframes = [self.df_train, self.df_val]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)


class EnzPredDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_file = Path(data_dir).with_suffix(".csv")
        self._data_stem = Path(self._data_file.stem)
        self._data_dir = self._data_file.parent / self._data_file.stem
        self._seed = 0
        self._replicate = seed

        df = pd.read_csv(self._data_file, index_col=0)
        self._drug_column = df.columns[1]
        self._target_column = df.columns[0]
        self._label_column = df.columns[2]

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    @classmethod
    def dataset_list(cls):
        return [
            "halogenase",
            "bkace",
            "gt",
            "esterase",
            "kinase",
            "phosphatase",
        ]

    def prepare_data(self):
        os.makedirs(self._data_dir, exist_ok=True)

        kfsplitter = KFold(n_splits=10, shuffle=True, random_state=self._seed)
        full_data = pd.read_csv(self._data_file, index_col=0)

        all_drugs = full_data[self._drug_column].unique()
        all_targets = full_data[self._target_column].unique()

        if self.drug_featurizer.path.exists() and self.target_featurizer.path.exists():
            logg.warning("Drug and target featurizers already exist")

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        if not self.drug_featurizer.path.exists():
            self.drug_featurizer.write_to_disk(all_drugs)

        if not self.target_featurizer.path.exists():
            self.target_featurizer.write_to_disk(all_targets)

        self.drug_featurizer.cpu()
        self.target_featurizer.cpu()

        for i, split in enumerate(kfsplitter.split(full_data)):
            fold_train = full_data.iloc[split[0]].reset_index(drop=True)
            fold_test = full_data.iloc[split[1]].reset_index(drop=True)
            logg.debug(self._data_dir / self._data_stem.with_suffix(f".{i}.train.csv"))
            fold_train.to_csv(
                self._data_dir / self._data_stem.with_suffix(f".{i}.train.csv"),
                index=True,
                header=True,
            )
            fold_test.to_csv(
                self._data_dir / self._data_stem.with_suffix(f".{i}.test.csv"),
                index=True,
                header=True,
            )

    def setup(self, stage: T.Optional[str] = None):
        df_train = pd.read_csv(
            self._data_dir
            / self._data_stem.with_suffix(f".{self._replicate}.train.csv"),
            index_col=0,
        )
        self.df_train, self.df_val = train_test_split(df_train, test_size=0.1)
        self.df_test = pd.read_csv(
            self._data_dir
            / self._data_stem.with_suffix(f".{self._replicate}.test.csv"),
            index_col=0,
        )

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = BinaryDataset(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

            self.data_val = BinaryDataset(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

        if stage == "test" or stage is None:
            self.data_test = BinaryDataset(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                self.drug_featurizer,
                self.target_featurizer,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)


class DUDEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task_dir: str,
        contrastive_split: str,
        drug_featurizer: Featurizer,
        target_featurizer: Featurizer,
        device: torch.device = torch.device("cpu"),
        n_neg_per: int = 50,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=None,
        sep="\t",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": contrastive_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device
        self._n_neg_per = n_neg_per

        self._data_dir = task_dir
        self._split = contrastive_split
        self._split_path = self._data_dir / Path(
            f"dude_{self._split}_type_train_test_split.csv"
        )

        self._drug_id_column = "Molecule_ID"
        self._drug_column = "Molecule_SMILES"
        self._target_id_column = "Target_ID"
        self._target_column = "Target_Seq"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def prepare_data(self):
        pass

    #         self.df_full = pd.read_csv(
    #             self._data_dir / Path("full.tsv"), **self._csv_kwargs
    #         )
    #         all_drugs = self.df_full[self._drug_column].unique()
    #         all_targets = self.df_full[self._target_column].unique()

    #         if self._device.type == "cuda":
    #             self.drug_featurizer.cuda(self._device)
    #             self.target_featurizer.cuda(self._device)

    #         self.drug_featurizer.write_to_disk(all_drugs)
    #         self.target_featurizer.write_to_disk(all_targets)

    def setup(self, stage: T.Optional[str] = None):
        self.df_full = pd.read_csv(
            self._data_dir / Path("full.tsv"), **self._csv_kwargs
        )

        self.df_splits = pd.read_csv(self._split_path, header=None)
        self._train_list = self.df_splits[self.df_splits[1] == "train"][0].values
        self._test_list = self.df_splits[self.df_splits[1] == "test"][0].values

        self.df_train = self.df_full[
            self.df_full[self._target_id_column].isin(self._train_list)
        ]
        self.df_test = self.df_full[
            self.df_full[self._target_id_column].isin(self._test_list)
        ]

        self.train_contrastive = make_contrastive(
            self.df_train,
            self._drug_column,
            self._target_column,
            self._label_column,
            self._n_neg_per,
        )

        self._dataframes = [self.df_train]  # , self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(self._device)
            self.target_featurizer.cuda(self._device)

        self.drug_featurizer.preload(all_drugs, write_first=True)
        self.drug_featurizer.cpu()

        self.target_featurizer.preload(all_targets, write_first=True)
        self.target_featurizer.cpu()

        if stage == "fit" or stage is None:
            self.data_train = ContrastiveDataset(
                self.train_contrastive["Anchor"],
                self.train_contrastive["Positive"],
                self.train_contrastive["Negative"],
                self.drug_featurizer,
                self.target_featurizer,
            )

        # if stage == "test" or stage is None:
        #     self.data_test = BinaryDataset(self.df_test[self._drug_column],
        #                                     self.df_test[self._target_column],
        #                                     self.df_test[self._label_column],
        #                                     self.drug_featurizer,
        #                                     self.target_featurizer
        #                                    )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)


#     def val_dataloader(self):
#         return DataLoader(self.data_test,
#                         **self._loader_kwargs
#                          )

#     def test_dataloader(self):
#         return DataLoader(self.data_test,
#                          **self._loader_kwargs
#                          )
