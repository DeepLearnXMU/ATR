dict(
    # network structure
    dim_word=300,  # word vector dimensionality
    dim=300,      # the number of hidden units
    word_embedding='./snli_1.0/vector.pkl',

    # process control
    max_epochs=20,
    finish_after=100000000,  # finish after this many updates
    dispFreq=1,
    saveto='snli_atr_model.npz',
    validFreq=6000,  # tunable
    saveFreq=1000,   # save the parameters after every saveFreq updates
    reload_=True,
    overwrite=True,

    # layer type
    layer='atr',
    dprate=0.85,

    # optimization
    decay_c=0.,  # L2 regularization penalty
    clip_c=1.,   # gradient clipping threshold
    lrate=5e-4,  # learning rate
    optimizer='adam',
    batch_size=128,
    valid_batch_size=128,
    seed=123,   # tunable

    # datasets
    train_data='./snli_1.0/train.pkl',
    dev_data='./snli_1.0/dev.pkl',
    test_data='./snli_1.0/test.pkl'
)
