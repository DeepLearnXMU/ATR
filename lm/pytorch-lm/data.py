# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


class Dataset(object):
    def __init__(self, txt_file, vocab):
        self.text = txt_file
        self.vocab = vocab

    def load_data(self):
        with open(self.text, 'r') as reader:
            data_text = reader.read().replace("\n", "<eos>").split()
            data_ids = self.vocab.to_id(data_text, append_eos=False)
            return data_ids

    def batcher(self, batch_size, num_steps):
        dataset = np.array(self.load_data(), dtype=np.int32)

        batch_len = len(dataset) // batch_size
        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            data[i] = dataset[i * batch_len: (i+1) * batch_len]

        epoch_size = (batch_len - 1) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size is 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i * num_steps: (i+1) * num_steps]
            y = data[:, i * num_steps + 1: (i+1) * num_steps + 1]
            yield (torch.LongTensor(x), torch.LongTensor(y))
