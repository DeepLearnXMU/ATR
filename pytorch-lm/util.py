# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch


def apply_dropout(tensor, p, is_train):
    # p: dropout value
    # why use self-implemented dropout, this is because
    # dropout with drop(x, p) / (1 - p) cause instability
    # another trap compared with tensorflow.
    if is_train:
        binary_mask = tensor.new_tensor(
            torch.rand(tensor.size()) > p)
        return tensor * binary_mask
    else:
        return tensor * (1. - p)
