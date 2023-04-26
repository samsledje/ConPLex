from types import SimpleNamespace

import os
import pickle as pk
from functools import lru_cache

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..featurizer.protein import FOLDSEEK_MISSING_IDX
from ..utils import get_logger

logg = get_logger()

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}

#######################
# Model Architectures #
#######################


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]), requires_grad=False)
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
        ).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.
        :meta private:
        """
        self.k.data.clamp_(min=0)


#######################
# Model Architectures #
#######################


class SimpleCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


SimpleCoembeddingNoSigmoid = SimpleCoembedding


class SimpleCoembeddingSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCoembedding_FoldSeek(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=1024,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class SimpleCoembedding_FoldSeekX(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=512,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        # self.projector_dropout = nn.Dropout(p=0.2)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class GoldmanCPI(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=100,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        model_dropout=0.2,
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        self.last_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, 1, bias=True),
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        output = torch.einsum("bd,bd->bd", drug_projection, target_projection)
        distance = self.last_layers(output)
        return distance

    def classify(self, drug, target):
        distance = self.regress(drug, target)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class AffinityCoembedInner(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.mol_projector[0].weight)

        print(self.mol_projector[0].weight)

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.prot_projector[0].weight)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        print(mol_proj)
        print(prot_proj)
        y = torch.bmm(
            mol_proj.view(-1, 1, self.latent_size),
            prot_proj.view(-1, self.latent_size, 1),
        ).squeeze()
        return y


class CosineBatchNorm(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, self.latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, self.latent_size),
            latent_activation(),
        )

        self.mol_norm = nn.BatchNorm1d(self.latent_size)
        self.prot_norm = nn.BatchNorm1d(self.latent_size)

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_norm(self.mol_projector(mol_emb))
        prot_proj = self.prot_norm(self.prot_projector(prot_emb))

        return self.activator(mol_proj, prot_proj)


class LSTMCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        lstm_layers=3,
        lstm_dim=256,
        latent_size=256,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.rnn = nn.LSTM(
            self.prot_emb_size,
            lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(2 * lstm_layers * lstm_dim, latent_size), nn.ReLU()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)

        outp, (h_out, _) = self.rnn(prot_emb)
        prot_hidden = h_out.permute(1, 0, 2).reshape(outp.shape[0], -1)
        prot_proj = self.prot_projector(prot_hidden)

        return self.activator(mol_proj, prot_proj)


class DeepCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        hidden_size=4096,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, hidden_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
            nn.Linear(hidden_size, latent_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class SimpleConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        hidden_dim_1=512,
        hidden_dim_2=256,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.fc1 = nn.Sequential(
            nn.Linear(mol_emb_size + prot_emb_size, hidden_dim_1), activation()
        )
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), activation())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim_2, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc3(self.fc2(self.fc1(cat_emb))).squeeze()


class SeparateConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric=None,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.fc = nn.Sequential(nn.Linear(2 * latent_size, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


class AffinityEmbedConcat(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )

        self.fc = nn.Linear(2 * latent_size, 1)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


SimplePLMModel = AffinityEmbedConcat


class AffinityConcatLinear(nn.Module):
    def __init__(
        self,
        mol_emb_size,
        prot_emb_size,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.fc = nn.Linear(mol_emb_size + prot_emb_size, 1)

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc(cat_emb).squeeze()
