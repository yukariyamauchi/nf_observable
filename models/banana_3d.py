from dataclasses import dataclass
from typing import Tuple
import numpy as np
import jax.numpy as jnp

@dataclass
class Model:
    a: float
    b: float
    c: float
    dim = 3

    # returns the log of the distribution
    def dist(self, x):
        return -(self.a-x[0])**2 - self.b*(x[1]-x[0]**2)**2 - self.c*(x[2]-x[1]**2-0.2*x[0]**2)**2

    def observables(self, x):
        return jnp.array([x[0]**2, x[1]*x[2]])

