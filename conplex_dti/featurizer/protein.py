import typing as T

import hashlib
import os
import pickle as pk
import sys
from pathlib import Path

import torch

from ..utils import get_logger
from .base import Featurizer

logg = get_logger()

MODEL_CACHE_DIR = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
)
FOLDSEEK_DEFAULT_PATH = Path(
    "/afs/csail.mit.edu/u/r/rsingh/work/corals/data-scratch1/ConPLex_foldseek_embeddings/r1_foldseekrep_encoding.p"
)
FOLDSEEK_MISSING_IDX = 20

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


class BeplerBergerFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("BeplerBerger", 6165, save_dir)
        from dscript.language_model import lm_embed

        self._max_len = 800
        self._embed = lm_embed

    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[: self._max_len]

        lm_emb = self._embed(seq, use_cuda=self.on_cuda)
        return lm_emb.squeeze().mean(0)


class ESMFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("ESM", 1280, save_dir)

        import esm

        torch.hub.set_dir(MODEL_CACHE_DIR)

        self._max_len = 1024

        (
            self._esm_model,
            self._esm_alphabet,
        ) = esm.pretrained.esm1b_t33_650M_UR50S()
        self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
        self._register_cuda("model", self._esm_model)

    def _transform(self, seq: str):
        seq = seq.upper()
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        batch_labels, batch_strs, batch_tokens = self._esm_batch_converter(
            [("sequence", seq)]
        )
        batch_tokens = batch_tokens.to(self.device)
        results = self._cuda_registry["model"][0](
            batch_tokens, repr_layers=[33], return_contacts=True
        )
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        tokens = token_representations[0, 1 : len(seq) + 1]

        return tokens.mean(0)


# class ProseFeaturizer(Featurizer):
#     def __init__(self, save_dir: Path = Path().absolute(), per_tok=False):
#         super().__init__("Prose", 6165, save_dir)

#         from prose.alphabets import Uniprot21
#         from prose.models.multitask import ProSEMT

#         self._max_len = 800
#         self.per_tok = per_tok
#         self._prose_model = ProSEMT.load_pretrained(
#             path=f"{MODEL_CACHE_DIR}/prose_mt_3x1024.sav"
#         )

#         self._register_cuda("model", self._prose_model)

#         self._prose_alphabet = Uniprot21()

#     def _transform(self, seq):
#         if len(seq) > self._max_len:
#             seq = seq[: self._max_len]

#         x = seq.upper().encode("utf-8")
#         x = self._prose_alphabet.encode(x)
#         x = torch.from_numpy(x)
#         x = x.to(self.device)
#         x = x.long().unsqueeze(0)

#         z = self._cuda_registry["model"][0].transform(x)
#         z = z.squeeze(0)

#         if self.per_tok:
#             return z
#         return z.mean(0)


class ProtBertFeaturizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False):
        super().__init__("ProtBert", 1024, save_dir)

        from transformers import AutoModel, AutoTokenizer, pipeline

        self._max_len = 1024
        self.per_tok = per_tok

        self._protbert_tokenizer = AutoTokenizer.from_pretrained(
            "Rostlab/prot_bert",
            do_lower_case=False,
            cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers",
        )
        self._protbert_model = AutoModel.from_pretrained(
            "Rostlab/prot_bert",
            cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers",
        )
        self._protbert_feat = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
        )

        self._register_cuda("model", self._protbert_model)
        self._register_cuda("featurizer", self._protbert_feat, self._feat_to_device)

    def _feat_to_device(self, pipe, device):
        from transformers import pipeline

        if device.type == "cpu":
            d = -1
        else:
            d = device.index

        pipe = pipeline(
            "feature-extraction",
            model=self._protbert_model,
            tokenizer=self._protbert_tokenizer,
            device=d,
        )
        self._protbert_feat = pipe
        return pipe

    def _space_sequence(self, x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        embedding = torch.tensor(
            self._cuda_registry["featurizer"][0](self._space_sequence(seq))
        )
        seq_len = len(seq)
        start_Idx = 1
        end_Idx = seq_len + 1
        feats = embedding.squeeze()[start_Idx:end_Idx]

        if self.per_tok:
            return feats
        return feats.mean(0)


class ProtT5XLUniref50Featurizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute(), per_tok=False):
        super().__init__("ProtT5XLUniref50", 1024, save_dir)

        self._max_len = 1024
        self.per_tok = per_tok

        (
            self._protbert_model,
            self._protbert_tokenizer,
        ) = ProtT5XLUniref50Featurizer._get_T5_model()
        self._register_cuda("model", self._protbert_model)

    @staticmethod
    def _get_T5_model():
        from transformers import T5EncoderModel, T5Tokenizer

        model = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers",
        )
        model = model.eval()  # set model to evaluation model
        tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/prot_t5_xl_uniref50",
            do_lower_case=False,
            cache_dir=f"{MODEL_CACHE_DIR}/huggingface/transformers",
        )

        return model, tokenizer

    @staticmethod
    def _space_sequence(x):
        return " ".join(list(x))

    def _transform(self, seq: str):
        if len(seq) > self._max_len - 2:
            seq = seq[: self._max_len - 2]

        token_encoding = self._protbert_tokenizer.batch_encode_plus(
            ProtT5XLUniref50Featurizer._space_sequence(seq),
            add_special_tokens=True,
            padding="longest",
        )
        input_ids = torch.tensor(token_encoding["input_ids"])
        attention_mask = torch.tensor(token_encoding["attention_mask"])

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            embedding = self._cuda_registry["model"][0](
                input_ids=input_ids, attention_mask=attention_mask
            )
            embedding = embedding.last_hidden_state
            seq_len = len(seq)
            start_Idx = 1
            end_Idx = seq_len + 1
            seq_emb = embedding[0][start_Idx:end_Idx]

        return seq_emb.mean(0)


class CNN2Layers(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        feature_channels,
        kernel_size,
        stride,
        padding,
        dropout,
    ):
        super(CNN2Layers, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=feature_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(
                in_channels=feature_channels,
                out_channels=3,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        x = self.conv1(x)
        return torch.squeeze(x)


class BindPredict21Featurizer(Featurizer):
    def __init__(self, save_dir: Path = Path().absolute()):
        super().__init__("BindPredict21", 128, save_dir)

        self._model_path = f"{MODEL_CACHE_DIR}/bindpredict_saved/checkpoint5.pt"

        self._max_len = 1024

        _p5tf = ProtT5XLUniref50Featurizer(save_dir=save_dir)
        _md = CNN2Layers(1024, 128, 5, 1, 2, 0)
        _md.load_state_dict(
            torch.load(self._model_path, map_location="cpu")["state_dict"]
        )
        _md = _md.eval()
        _cnn_first = _md.conv1[:2]

        self._register_cuda("pt5_featurizer", _p5tf)
        self._register_cuda("model", _md)
        self._register_cuda("cnn", _cnn_first)

    def _transform(self, seq):
        if len(seq) > self._max_len:
            seq = seq[: self._max_len]

        protbert_e = self._cuda_registry["pt5_featurizer"][0](seq)
        bindpredict_e = self._cuda_registry["cnn"][0](protbert_e.view(1, 1024, -1))
        return bindpredict_e.mean(axis=2).squeeze()


# class DSCRIPTFeaturizer(Featurizer):
#     def __init__(self, save_dir: Path = Path().absolute()):
#         super().__init__("DSCRIPT", 100, save_dir)

#         self._max_len = 800

#         self._dscript_model = get_pretrained("human_v1")
#         self._register_cuda(
#             "model", self._dscript_model, DSCRIPTFeaturizer._dscript_to_device
#         )

#     @staticmethod
#     def _dscript_to_device(model, device: torch.device):
#         model = model.to(device)
#         model.use_cuda = device.type == "cuda"
#         return model

#     def _transform(self, seq: str):
#         if len(seq) > self._max_len:
#             seq = seq[: self._max_len]

#         with torch.set_grad_enabled(False):
#             lm_emb = lm_embed(seq, use_cuda=(self.device.type == "cuda"))
#             lm_emb = lm_emb.to(self.device)
#             ds_emb = self._cuda_registry["model"][0].embedding(lm_emb)
#             return ds_emb.mean(axis=1).squeeze()


class FoldSeekFeaturizer(Featurizer):
    def __init__(
        self,
        save_dir: Path = Path().absolute(),
        foldseek_pickle_path: T.Optional[Path] = None,
    ):
        super().__init__("FoldSeek", 22, save_dir)

        self._max_len = None

        if foldseek_pickle_path is not None:
            self._foldseek_pickle_path = foldseek_pickle_path
        else:
            self._foldseek_pickle_path = FOLDSEEK_DEFAULT_PATH

        self.pk_fs_dict = {}
        with open(self._foldseek_pickle_path, "rb") as pk_fi:
            fs_np_arrays = pk.load(pk_fi)
        for fs_tuple in fs_np_arrays:
            key, afkey, embedding = fs_tuple
            key_hash = key.split("_")[-1]
            self.pk_fs_dict[key_hash] = torch.from_numpy(embedding)

    @staticmethod
    def _md5_hex_hash(seq: str):
        return hashlib.md5(seq.encode("utf-8")).hexdigest()

    @staticmethod
    def _default_missing_foldseek_embedding(seq):
        seqlen = len(seq)
        return FOLDSEEK_MISSING_IDX * torch.ones(seqlen)

    def _transform(self, seq: str):
        seq_hash = FoldSeekFeaturizer._md5_hex_hash(seq)
        fs_embedding = self.pk_fs_dict.setdefault(
            seq_hash,
            FoldSeekFeaturizer._default_missing_foldseek_embedding(seq),
        )
        return fs_embedding
