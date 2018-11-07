This is our source code for chinese-word-segmentation experiments.

* Requirement
    - Theano
    - Numpy

* Unfortunately, we can not release the dataset.

#### How to Run?
1. configure theano gpu environment, e.g. GPU index
2. see `run.sh` in `atr` folder.

During our experiments, we manually fine-tuned the random seed, validation training steps etc. to get
a better score for ATR model.