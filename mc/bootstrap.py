#!/usr/bin/env python

import numpy as np
import sys
from cmath import phase

def bootstrap(xs, B, ws=None, N=100):
    if B > len(xs):
        print('block size must be smaller than the number of samples')
        exit()
    Bs = len(xs)//B
    if ws is None:
        ws = xs*0 + 1
    # Block
    x, w = [], []
    for i in range(Bs):
        x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
        w.append(sum(ws[i*B:i*B+B]))
    x = np.array(x)
    w = np.array(w)
    # Regular bootstrap
    y = x * w
    m = (sum(y) / sum(w))
    ms = []
    for n in range(N):
        s = np.random.choice(range(len(x)), len(x))
        ms.append((sum(y[s]) / sum(w[s])))
    ms = np.array(ms)
    return m, np.std(ms.real)

if __name__ == '__main__':
    filename = sys.argv[1]
    b = int(sys.argv[2])

    dat = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            r = [np.float64(x) for x in l.split()]
            dat.append(r)

    dat = np.array(dat)

    for i in range(dat.shape[1]):
        mean, std = bootstrap(dat[:,i], b)
        print(f'{i}th observable: mean {mean:.5f} std {std:.5f}')

