#!/usr/bin/env python

# Libraries
from functools import partial
import pickle
import sys
from typing import Callable
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models import banana_3d
from mc import metropolis

# Specify to use CPU, not GPU.
jax.config.update('jax_platform_name', 'cpu')

#def mcmc(model, n, skip, seed=None):
def mcmc(model, filename, nsamples, skip, seed=None):
    '''
    MCMC samples from the distribution in model file
    Args:
        model (str): Model file.
        filename (str): File to store samples.
        nsamples (int): Number of samples.
        skip (int): Number of MCMC steps to skip per sample.
        seed (int): Random seed/
    '''


    jax.config.update("jax_debug_nans", True)

    with open(model, 'rb') as f:
        model = eval(f.read())
    D = model.dim

    if seed == None:
        seed = time.time_ns()
    chainKey = jax.random.PRNGKey(seed)

    # Thermalization steps
    Ntherm = 1000

    # Generate samples via MCMC chain
    chain = metropolis.Chain(model.dist, jnp.zeros(D), chainKey)
    chain.step(N=Ntherm)
    chain.calibrate()

    if filename == None:   
        for i in range(nsamples):
            chain.step(N=skip)
            str_sample_dist = ','.join([str(x) for x in chain.x] + [str(model.dist(chain.x))] + [str(model.observables(chain.x))])
            print(str_sample_dist)
    else:
        with open(filename, 'w') as f:
            for i in range(nsamples):
                chain.step(N=skip)
                f.write(','.join([str(x) for x in chain.x] + [str(model.dist(chain.x))]) + '\n')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description="MCMC sampling",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('model', type=str, help="model filename")
    parser.add_argument('-c', '--skip', type=int, default=1, help="number of samples skipped in MCMC")
    parser.add_argument('-f', '--filename', type=str, help="file to save the data")
    parser.add_argument('-n', '--nsamples', type=int, default=1000, help="total number of samples")
    parser.add_argument('--seed', type=int, help="random seed for sampling")
    args = parser.parse_args()

    mcmc(args.model, args.filename, args.nsamples, args.skip, args.seed)
