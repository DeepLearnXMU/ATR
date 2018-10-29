# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from torch.nn.utils.rnn import \
    pack_padded_sequence, pad_packed_sequence

import util


class LSTM(nn.Module):
    """LSTM layer with dropout"""

    def __init__(self,
                 input_size,        # input dimension,
                 hidden_size,       # hidden or output dimension
                 dropout=0.0,       # dropout rate
                 ):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        """Parameter Initialization, also you can safely ignore it"""
        # bias not init with zero, because case nan for some architecture
        self.i2h.bias.data.uniform_(-0.08, 0.08)
        self.h2h.bias.data.uniform_(-0.08, 0.08)

        self.i2h.weight.data.uniform_(-0.08, 0.08)
        self.h2h.weight.data.uniform_(-0.08, 0.08)

    def forward(self, inputs, init_state=None):
        # inputs: packed sequence with shape [batch, length, dim]
        # init_state: [1, batch, dim]
        # just valid for one-layer lstm

        seq_tensor, seq_lengths = pad_packed_sequence(inputs,
                                                      batch_first=True)
        batch_size, time_steps = seq_tensor.size()[:2]

        if init_state is None:
            prev_h = seq_tensor.new_zeros(batch_size, self.hidden_size)
            prev_c = seq_tensor.new_zeros(batch_size, self.hidden_size)
        else:
            prev_h = init_state[0].squeeze(0)
            prev_c = init_state[1].squeeze(0)

        valid_batch_index = batch_size - 1
        outputs = seq_tensor.new_zeros(batch_size, time_steps, self.hidden_size)

        for step in range(time_steps):

            index = step

            # 1. generate the valid batch index
            while seq_lengths[valid_batch_index] <= index:
                valid_batch_index -= 1

            # 2. extract rnn input and previous states
            t = valid_batch_index + 1
            x = seq_tensor[:t, index]
            h_ = prev_h[:t].clone()
            c_ = prev_c[:t].clone()

            preact = self.i2h(x) + self.h2h(h_)

            i, f, o, g = preact.split(self.hidden_size, dim=-1)

            c = c_ * f.sigmoid() + i.sigmoid() * g.tanh()

            h = o.sigmoid() * c.tanh()

            prev_h = prev_h.data.clone()
            prev_c = prev_c.data.clone()
            prev_h[:t] = h
            prev_c[:t] = c
            outputs[:t, index] = h

        outputs = util.apply_dropout(outputs, self.dropout, self.training)
        outputs = pack_padded_sequence(outputs, seq_lengths, batch_first=True)

        # [num_layer * direction, batch, dim]
        final_state = (prev_h.unsqueeze(0), prev_c.unsqueeze(0))

        return outputs, final_state
