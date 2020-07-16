# ATR ---- A novel recurrent unit for RNN.

> Fork from [ATR](https://github.com/bzhangGo/ATR)

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
model capacity. For more details, please see our [Zero](https://github.com/bzhangGo/zero)
system.

Throughout our experiments, we mainly use shallow RNN models.
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

## Optional Designs
- You can also apply non-linearity to the hidden states, which sometimes can improve
the performance.

  For example, 
  * you can change `h = i * p + f * h_prev` to `h = tanh(i * p + f * h_prev))`; or
  * you can change `h = i * p + f * h_prev` to `h = i * tanh(p) + f * h_prev`; or
  * you can just apply `tanh` to `h` for output: `o = tanh(h)` and use `o` for 
  the following models.
- You can also replace the input gate with `1. - f` in a general manner.  

*Again, these desings are not necessary.*

## How to use this repository?

See `lm`, `snli`, `cws` for language model, natural language inference and chinese word
segmentation respectively.

The `lm` part is implemented in tensorflow and pytorch, while others are implemented in 
theano.

## TODO
* Implement the ATR unit in cuda-level. Welcome contributions in this direction.

## Contact
For any question, please feel free to contact [Biao Zhang](mailto:B.Zhang@ed.ac.uk).
