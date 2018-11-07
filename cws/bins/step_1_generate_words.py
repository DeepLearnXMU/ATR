#! /usr/bin/python

import sys

plain_char = sys.argv[1]
pred_tag = sys.argv[2]
output = sys.argv[3]

def read(f):
    d = []
    for line in file(f, 'rU'):
        d.append(line.strip().split())
    return d

charsen = read(plain_char)
tagsen = read(pred_tag)
assert len(charsen) == len(tagsen)

out = file(output, 'w')
for char, tag in zip(charsen, tagsen):
    if len(tag) - len(char) == 1:
        tag = tag[:-1]
    assert len(char) == len(tag)

    words = []
    word = ''
    for c, t in zip(char, tag):
        if t == 'B': # a new word
            if word != '':
                words.append(word.strip())
            word = c
        elif t == 'M': # continue
            word += c
        elif t == 'E': # a new word finished
            word += c
            words.append(word.strip())
            word = ''
        elif t == 'S':
            if word != '':
                words.append(word.strip())
            words.append(c)
            word = ''
        else: # default as S
            if word != '':
                words.append(word.strip())
            words.append(c)
            word = ''
    if word != '':
        words.append(word.strip())
    print >> out, ' '.join(words)
out.close()
