dict(
        # network structure
        dim_word=620,  # word vector dimensionality
        dim=1000,      # the number of LSTM units
        encoder='gru',
        decoder='gru_cond',
        n_words_src=30000,  # source vocabulary size
        n_words=30000,  # target vocabulary size
        maxlen=50,  # maximum length of the description

        # process control
        max_epochs=10,
        finish_after=100000000,  # finish after this many updates
        dispFreq=1,
        saveto='search_model.npz',
        validFreq=5000,
        validFreqLeast=10000,
        validFreqFires=10000,
        validFreqRefine=1000,
        saveFreq=1000,   # save the parameters after every saveFreq updates
        sampleFreq=1000,   # generate some samples after every sampleFreq
        reload_=True,
        overwrite=False,
        is_eval_nist=True,

        # optimization
        decay_c=0.,  # L2 regularization penalty
        alpha_c=0.,  # alignment regularization
        clip_c=1.,   # gradient clipping threshold
        lrate=0.0001,  # learning rate
        optimizer='adadelta',
        batch_size=32,
        valid_batch_size=32,
        use_dropout=False,
        shuffle_train=1.,
        seed=1234,

        # development evaluation
        use_bleueval=True,
        save_devscore_to='search_score.log',
        save_devtrans_to='search_trans.txt',
        beam_size=10,
        proc_num=5,
        normalize=False,
        output_nbest=1,

        # datasets
        datasets=[
            '/home/bzhang/.pro/.lib/neumt/github/nmt-master/nmt/works/step_-1_prepare_datas/corpus.zh-en.zh',
            '/home/bzhang/.pro/.lib/neumt/github/nmt-master/nmt/works/step_-1_prepare_datas/corpus.zh-en.en'],
        # for valiadation
        # default, the MOSES evaluation inputs: plain source and plain target
        # the NIST evaluation inputs: plain source and sgm target, we treat the "source file name".sgm as the sgm source
        valid_datasets=['/home/bzhang/.pro/.cps/nist_test/nist05/src.plain',
                        '/home/bzhang/.pro/.cps/nist_test/nist05/ref.sgm'],
        dictionaries=[
            '/home/bzhang/.pro/.lib/neumt/github/nmt-master/nmt/works/step_-1_prepare_datas/vocab.zh.pkl',
            '/home/bzhang/.pro/.lib/neumt/github/nmt-master/nmt/works/step_-1_prepare_datas/vocab.en.pkl'],
)
