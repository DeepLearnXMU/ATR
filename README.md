# ATR ---- A novel recurrent unit for RNN.

ATR denotes: Addition-Subtraction Twin-Gated Recurrent Unit.
It relies on a twin gate mechanism that utilizes
addition and subtraction operation to generate an
input gate and a forget gate respectively.

The idea is proposed in our EMNLP18 conference paper:
`Simplifying Neural Machine Translation with Addition-Subtraction
Twin-Gated Recurrent Networks`. If you use ATR cell, please consider
citing it:
```
@InProceedings{D18-1459,
  author = 	"Zhang, Biao
		and Xiong, Deyi
		and su, jinsong
		and Lin, Qian
		and Zhang, Huiji",
  title = 	"Simplifying Neural Machine Translation with Addition-Subtraction Twin-Gated Recurrent Networks",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"4273--4283",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1459"
}
```

To help others quickly learn the structure, we 
implement a word-based LM model on PTB dataset using
both tensorflow and pytorch framework. We didnot pay
too much efforts to optimize the hyperparameters. By
LM task, we aim to show how the model works and give
an intuitive comparison among LSTM, GRU and ATR.

Our main application of ATR is the machine translation
task, where source sentence semantic encoding, target
conditional language modeling, and complex semantic
reasoning to capture source-target translation correspondence
are required. This is a very suitable platform for
testing RNN-based models, particularly in terms of
model capacity. For more details, please see our `Zero`
system.

Throughout our experiments, we only use shall RNN models.
Though recent researches could claim encouraging performance
with deep architectures, we still believe a shallow model
is the right way to demonstrate its capability.

## Architecture

```
Given current input x and previous hidden state h_rev, ATR composes them as follows:
\begin{equation}
p = W x
q = U h_prev
i = sigmoid(p + q)
f = sigmoid(p - q)
h = i * p + f * h_prev
\end{equation}
where W and U are the only weight matrices, just as a vanilla Elman structure.
```

> Notice that the computation of forget gate `f` is very important and sensitive,
you should subtract `x` related `p` with `h_prev` related `q`, i.e. p - q, so that 
the previous hidden state can help adjust the forget gate value to avoid value 
explosion. The equations above are slightly different from those in our paper, 
mainly on the computation of the forget gate `f` where we made a typo in the paper :(.

> In short, these two gates are order-sensitive!

## How to use this repository?

### To use the pytorch-based model

1. requirement

* pytorch: 0.4.1 (the tested version)
* numpy

2. training script

```
data_dir=path-to-the-data-dir
CUDA_VISIBLE_DEVICES=0 python pytorch-lm/main.py --cuda --data_dir  $data_dir\
    --vocab_dir $data_dir/vocab.txt \
    --cell ATR
```

### To use the tensrflow-based model

1. requirement

 * tensorflow: > 0.8.0
 * numpy
 
2. training script

```
data_dir=path-to-the-data-dir
CUDA_VISIBLE_DEVICES=0 python pytorch-lm/main.py --data_dir=$data_dir
```

## TODO
* Implement the ATR unit in cuda-level. Welcome contributions in this direction.

## Contact
For any question, please free to contact the [first author](B.Zhang@ed.ac.uk).