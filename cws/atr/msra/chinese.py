dict(
        # network structure 
        dim_word=300,  # word vector dimensionality
        dim=300,      # the number of LSTM units
        encoder='atr',
        decoder='atr_cond',
        n_words_src=5033,  # source vocabulary size
        n_tags=4,  # target vocabulary size
        maxlen=500,  # maximum length of the description

        ita=0.2,

        # process control
        max_epochs=50,
        finish_after=100000000,  # finish after this many updates
        dispFreq=1,
        saveto='search_model.npz',
        validFreq=100,
        validFreqLeast=0,
        validFreqFires=20000,
        validFreqRefine=1000,
        saveFreq=1000,   # save the parameters after every saveFreq updates
        sampleFreq=1000,   # generate some samples after every sampleFreq
        reload_=True,
        overwrite=True,
        is_eval_nist=False,
        lr_start=40, # disable the lrate shrink

        # optimization
        decay_c=0.,  # L2 regularization penalty
        alpha_c=0.,  # alignment regularization
        clip_c=1.,   # gradient clipping threshold
        lrate=5e-4,   # learning rate
        optimizer='adam',
        batch_size=128,
        valid_batch_size=128,
        use_dropout=True,
        use_dropout_word=True,
        shuffle_train=0.999,
        seed=1234,

        # development evaluation
        use_bleueval=True,
        save_devscore_to='search_bleu.log',
        save_devtrans_to='search_trans.txt',
        beam_size=10,
        proc_num=1,
        normalize=False,
        output_nbest=1,

        # datasets
        gold_vocab="./wdseger/prepare/datasets/preprocess/msr/msr_training.utf8",
        datasets=[
            './wdseger/prepare/datasets/preprocess/msr/train.utf8.rep.bmes.word',
            './wdseger/prepare/datasets/preprocess/msr/train.utf8.rep.bmes.tag'],
        valid_datasets=['./wdseger/prepare/datasets/preprocess/msr/dev.utf8.rep.bmes.word',
                        './wdseger/prepare/datasets/preprocess/msr/dev.utf8',
                        './wdseger/prepare/datasets/preprocess/msr/dev.utf8.back'],
        test_datasets=['./wdseger/prepare/datasets/preprocess/msr/test.utf8.rep.bmes.word',
                        './wdseger/prepare/datasets/preprocess/msr/test.utf8',
                        './wdseger/prepare/datasets/preprocess/msr/test.utf8.back'],
        dictionaries=[
            './wdseger/prepare/datasets/preprocess/msr/vocab.word.pkl',
            './wdseger/prepare/datasets/msr/vocab.tag2.pkl'],
)
