from functools import partial

import numpy as np
import torch
import torch.nn as nn


def tanh_decay(M_0, N_epoch, x):
    return M_0 * (1 - np.tanh(2 * x / N_epoch))


def cosine_anneal(M_0, N_epoch, x):
    return 0.5 * M_0 * (1 + np.cos(x * np.pi / N_epoch))


def no_decay(M_0, N_epoch, x):
    return M_0


def sigmoid_cosine_distance_p(x, y, p=1):
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p

def cosine_distance_p(x, y, p=1):
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - cosine_sim(x, y)) ** p

MARGIN_FN_DICT = {
    "tanh_decay": tanh_decay,
    "cosine_anneal": cosine_anneal,
    "no_decay": no_decay,
}

DIST_FN_DICT = {
    "sigmoid_cosine_distance": sigmoid_cosine_distance_p,
    "cosine_distance": cosine_distance_p,
}


class MarginScheduledLossFunction:
    def __init__(
        self,
        M_0: float = 0.25,
        N_epoch: float = 50,
        N_restart: float = -1,
        update_fn="tanh_decay",
        dist_fn="cosine_distance",
    ):
        self.M_0 = M_0
        self.N_epoch = N_epoch
        if N_restart == -1:
            self.N_restart = N_epoch
        else:
            self.N_restart = N_restart

        self._step = 0
        self.M_curr = self.M_0

        self._update_fn_str = update_fn
        self._update_margin_fn = self._get_update_fn(update_fn)
        self._dist_fn = DIST_FN_DICT[dist_fn]

        self._update_loss_fn()

    @property
    def margin(self):
        return self.M_curr

    def _get_update_fn(self, fn_string):
        return partial(MARGIN_FN_DICT[fn_string], self.M_0, self.N_restart)

    def _update_loss_fn(self):
        self._loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=sigmoid_cosine_distance_p,
            margin=self.M_curr,
        )

    def step(self):
        self._step += 1
        if self._step == self.N_restart:
            self.reset()
        else:
            self.M_curr = self._update_margin_fn(self._step)
            self._update_loss_fn()

    def reset(self):
        self._step = 0
        self.M_curr = self._update_margin_fn(self._step)
        self._update_loss_fn()

    def __call__(self, anchor, positive, negative):
        # logg.debug(anchor, anchor.shape)
        # logg.debug(positive, positive.shape)
        # logg.debug(negative, negative.shape)
        return self._loss_fn(anchor, positive, negative)
