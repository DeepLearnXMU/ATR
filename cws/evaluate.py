#! /usr/bin/python

'''
    Evaluate the translation result from Neural Machine Translation
'''

import os
run = os.system

bindir="./bins/"


def eval_trans(train_vocab, gold_word, plain_src, model_pred_tag, back_file=None):

    # step 1, generate pred_word
    run("%s/step_1_generate_words.py %s %s %s" % (bindir, plain_src, model_pred_tag, model_pred_tag+".words"))

    # step 1.1 replace with gold standard
    if back_file:
        run("%s/step_2_replace_back.py %s %s %s" % (bindir, model_pred_tag+".words", back_file, model_pred_tag+".words"+".good"))

    # step 2, scoring
    if not back_file:
        run("%s/score %s %s %s > %s.eval" % (bindir, train_vocab, gold_word, model_pred_tag+".words", model_pred_tag))
    else:
        run("%s/score %s %s %s > %s.eval" % (bindir, train_vocab, gold_word, model_pred_tag+".words"+".good", model_pred_tag))

    # step 3, extract F-score
    evals = file('%s.eval' % model_pred_tag, 'rU').readlines()
    results = evals[-1].split('\t')

    return float(results[-4])
