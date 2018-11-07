This is our source code for snli experiments.

* Requirement
    - Theano
    - Numpy

* Preprocessed SNLI dataset can be downloaded [here](https://drive.google.com/open?id=1BU2erdRrNCKfTzxZA30mrU4VjMcNEvUJ)

#### How to Run?
1. configure theano gpu environment, e.g. GPU index
2. train the model as `python -u train.py --proto=snli config.py`

Basically, you can get a Test Accuracy ~85. 
During our experiments, we manually fine-tuned the random seed, validation training steps etc. to get
a better acc for ATR model.