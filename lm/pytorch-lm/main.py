# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math

import torch
import argparse
import random
import numpy as np

from vocab import Vocab
from data import Dataset
from model import LanguageModel


def parse_args():
    parser = argparse.ArgumentParser(
        "Pytorch Word-Based Language Model")

    parser.add_argument("--data_dir", type=str,
                        help="The directory of datasets, including train, dev, and test")
    parser.add_argument("--vocab_dir", type=str,
                        help="The directory of vocabulary text")
    parser.add_argument("--embed_size", type=int, default=650,
                        help="The word embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=650,
                        help="The RNN hidden state size")
    parser.add_argument("--nlayers", type=int, default=2,
                        help="The RNN layers used for the model")
    parser.add_argument("--lr", type=float, default=20.0,
                        help="The learning rate")
    parser.add_argument("--clip", type=float, default=0.25,
                        help="The gradient clipping norm")
    parser.add_argument("--epochs", type=int, default=50,
                        help="The maximum training iterations")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="The batch size for one mini-batch")
    parser.add_argument("--disp_freq", type=int, default=100,
                        help="Display frequency")
    parser.add_argument("--num_steps", type=int, default=35,
                        help="The number of timesteps in one batch")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="The dropout value")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--cuda", action="store_true",
                        help="Use cuda")
    parser.add_argument("--save", type=str, default="model.pt",
                        help="Path to save the model")
    parser.add_argument("--cell", choices=["LSTM", "GRU", "ATR"],
                        default="ATR",
                        help="Recurrent cell type")

    return parser.parse_args()


def graph(params):

    model = LanguageModel(
        params.vocab.size(),
        params.embed_size,
        params.hidden_size,
        params.nlayers,
        dropout=params.dropout,
        cell=params.cell,
    )

    loss = torch.nn.CrossEntropyLoss()

    return model, loss


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    return tuple(v.detach() for v in h)


def eval(model, loss, dataset, params, batch_size=None):
    model.eval()

    if batch_size is None:
        batch_size = params.batch_size

    total_loss = 0.
    total_token = 0.
    hidden = model.init_hidden(batch_size)
    eval_data = Dataset(dataset, params.vocab)
    start_time = time.time()
    with torch.no_grad():
        for bidx, batch in enumerate(eval_data.batcher(
                batch_size,
                params.num_steps,
        )):
            x, t = batch

            x = x.to(params.device)
            t = t.to(params.device)

            hidden = repackage_hidden(hidden)
            logits, hidden = model(x, state=hidden)
            gloss = loss(
                logits.view(-1, logits.size(-1)), t.view(-1))

            num_token = (x > 0).float().sum()
            total_loss += gloss.item() * num_token
            total_token += num_token

    return total_loss / total_token, time.time() - start_time


def train(model, loss, params):
    train_data = Dataset(
        os.path.join(params.data_dir, "train.txt"),
        params.vocab,
    )

    total_loss = 0.
    global_step = 0
    start_time = time.time()
    best_valid_loss = None
    hidden = model.init_hidden(params.batch_size)
    lrate = params.lr
    for epoch in range(params.epochs):
        for bidx, batch in enumerate(train_data.batcher(
                params.batch_size,
                params.num_steps,
        )):
            model.train()

            x, t = batch

            x = x.to(params.device)
            t = t.to(params.device)

            hidden = repackage_hidden(hidden)
            model.zero_grad()
            logits, hidden = model(x, state=hidden)
            gloss = loss(
                logits.view(-1, logits.size(-1)), t.view(-1))

            gloss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params.clip)

            for p in model.parameters():
                p.data.add_(-lrate, p.grad.data)

            total_loss += gloss.item()

            if global_step > 0 \
                    and global_step % params.disp_freq == 0:
                sub_loss = total_loss / params.disp_freq
                duration = time.time() - start_time
                print('| Train | epoch {:3d} | {:5d} batches | '
                      'lr {:.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, bidx, lrate, duration * 1000 / params.disp_freq,
                    sub_loss, math.exp(sub_loss)))
                total_loss = 0.
                start_time = time.time()

            global_step += 1

        # start evaluation
        # keep the batch_size as default, since we do not need so
        # accurate batch_size
        score, speed = eval(model, loss,
                            os.path.join(params.data_dir, 'dev.txt'),
                            params)
        print('|  Dev  | epoch {:3d} | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, speed, score, math.exp(score)
        ))

        if not best_valid_loss or score < best_valid_loss:
            best_valid_loss = score
            with open(params.save, 'wb') as f:
                torch.save(model, f)
        else:
            lrate /= 4.0


if __name__ == "__main__":
    params = parse_args()

    # build vocabulary
    params.vocab = Vocab(vocab_file=params.vocab_dir)

    # set up random seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    # select device
    device = torch.device("cuda" if params.cuda else "cpu")
    params.device = device

    # build graph
    model, loss = graph(params)
    model = model.to(device)

    # start training
    train(model, loss, params)

    # start evaluation
    # After training, load best model
    with open(params.save, 'rb') as f:
        model = torch.load(f)

    # accurate evaluation
    score, speed = eval(model, loss,
                        os.path.join(params.data_dir, 'test.txt'),
                        params, batch_size=1)
    print('| End of Training, Test Result | ms/batch {:5.2f} | '
          'loss {:5.2f} | ppl {:8.2f}'.format(
        speed, score, math.exp(score))
    )
