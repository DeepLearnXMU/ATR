### The dataset

We provide the PTB dataset in the data folder.

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
CUDA_VISIBLE_DEVICES=0 python tensorflow-lm/main.py --data_path=$data_dir
```