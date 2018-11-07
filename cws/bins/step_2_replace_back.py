#! /usr/bin/python

import sys
reload(sys)
sys.setdefaultencoding('utf-8') 

words = file(sys.argv[1], 'rU').readlines()
backs = file(sys.argv[2], 'rU').readlines()
output = file(sys.argv[3], 'w')

for word, back in zip(words, backs):
    word = word.strip().decode('utf-8')
    back = back.strip().decode('utf-8')
    toks = word.strip().split()
    xs = back.strip().split()

    if len(xs) == 0:
        print >> output, word
    else:
        res = []
        xidx = 0
        for tok in toks:
            tmp = []
            for idx in xrange(len(tok)):
                if tok[idx] != 'X':
                    tmp.append(tok[idx])
                else:
                    tmp.append(xs[xidx])
                    xidx += 1
            res.append(''.join(tmp))
        assert xidx == len(xs)
        print >> output, ' '.join(res)
output.close()
