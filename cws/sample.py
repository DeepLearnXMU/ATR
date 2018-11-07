'''
Translates a source file using a translation model.
'''
import argparse

import numpy
import cPickle as pkl
import theano

from model import (build_model, load_params,
                 init_params, init_tparams, evaluation)

def main(model, dictionary, dictionary_tag, source_file, target_file, saveto):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    with open(dictionary, 'rb') as f:
        word_dict = pkl.load(f)
    word_idict = dict()
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # load tag dictionary and invert
    with open(dictionary_tag, 'rb') as f:
        tag_dict = pkl.load(f)
    tag_idict = dict()
    for kk, vv in tag_dict.iteritems():
        tag_idict[vv] = kk

    # allocate model parameters
    params = init_params(options)

    # load model parameters and set theano shared variables
    params = load_params(model, params)
    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost, predicts = \
        build_model(tparams, options)

    print 'Building f_predicts...',
    f_predicts = theano.function([x, x_mask], predicts)
    print 'Done'

    use_noise.set_value(0.)
    valid_err = evaluation(f_predicts, options, tag_idict, word_dict, source_file,
              saveto, target_file, 0, options['n_words_src'], back_file=target_file+".back")

    print 'Test ', valid_err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str)
    parser.add_argument('-dictionary', type=str)
    parser.add_argument('-dictionary_tag', type=str)
    parser.add_argument('-source', type=str)
    parser.add_argument('-target', type=str)
    parser.add_argument('-saveto', type=str)

    args = parser.parse_args()

    main(args.model, args.dictionary, args.dictionary_tag, args.source, args.target,
         args.saveto)
