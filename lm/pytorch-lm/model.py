# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.utils.rnn import \
    pack_padded_sequence, pad_packed_sequence

import util
from rnn import GRU, LSTM, ATR


class LanguageModel(nn.Module):
    """Simple word-based language model"""

    def __init__(self,
                 vocab_size,        # the vocabulary size
                 embed_size,        # word embedding size
                 hidden_size,       # hidden lstm size
                 num_layers,        # the number of RNN layer
                 dropout=0.0,       # variational dropout rate
                 cell='LSTM',       # rnn cell type
                 ):
        super(LanguageModel, self).__init__()
        self.cell = cell
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.project = hidden_size != embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = [eval(cell)(
            embed_size if l == 0 else hidden_size,
            hidden_size,
            dropout=dropout)
            for l in range(num_layers)]
        # make it on the attribute so that device can be
        # assigned automatically
        for lnum, layer in enumerate(self.rnn):
            self.add_module("rnn_{}_layer_{}".format(cell, lnum), layer)

        # by default, we share the embedding parameters
        # if the dimensions are mismatched, we solve it by another
        # linear layer
        if self.project:
            self.h2h = nn.Linear(hidden_size, embed_size)
        self.h2o = nn.Linear(embed_size, vocab_size)

        # share the embedding matrix and softmax matrix
        self.h2o.weight = self.embedding.weight

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.weight.data.uniform_(-0.08, 0.08)

        self.h2o.weight.data.uniform_(-0.08, 0.08)
        self.h2o.bias.data.zero_()

        if self.project:
            self.h2h.weight.data.uniform_(-0.08, 0.08)
            self.h2h.bias.data.zero_()

    def init_hidden(self, batch_size):
        weight = next((self.parameters()))

        if self.cell == "LSTM":
            hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                      weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),)

        return hidden

    def forward(self, inputs, lengths=None, state=None):

        if lengths is None:
            lengths = inputs.new_ones(inputs.size())
            lengths = lengths.sum(1)

        word_embed = self.embedding(inputs)
        word_embed = util.apply_dropout(word_embed, self.dropout, self.training)

        outputs = pack_padded_sequence(word_embed, lengths,
                                       batch_first=True)
        hs = []
        for l in range(self.num_layers):
            # extract states
            init_state = tuple(s[l:l+1] for s in state)
            # we simply stacking multiple rnn layers
            outputs, h = self.rnn[l](
                outputs, init_state=init_state)
            hs.append(h)

        outputs, lengths = pad_packed_sequence(outputs,
                                               batch_first=True)

        if self.project:
            outputs = self.h2h(outputs)
        logits = self.h2o(outputs)

        outputs_size = outputs.size()
        logits = logits.view(outputs_size[0], outputs_size[1],
                             logits.size(-1))

        state = tuple([torch.cat(vs, 0) for vs in list(zip(*hs))])

        return logits, state
