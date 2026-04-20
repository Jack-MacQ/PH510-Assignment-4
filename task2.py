"""
PH510 Assignment 5 - Task 2
Variational Monte Carlo study of the hydrogen 1s ground state.

This script applies the Variational Monte Carlo (VMC) method to the
hydrogen 1s radial problem using a trial wavefunction with variational
parameter alpha. 

This script extends the Task 1 VMC engine to carry out a systematic
parameter scan for ground-state optimisation.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt


ArrayFunc = Callable[[float], float]


@dataclass(frozen=True)
class VMCConfig:
    """Configuration parameters for a single VMC run."""

    n_samples: int = 100_000
    n_equilibration: int = 10_000
    decorrelation_steps: int = 5
    proposal_width: float = 1.0
    initial_position: float = 1.0
    seed: int = 42
    block_size: int = 100


@dataclass(frozen=True)
class VMCResult:
    """Results and sampling statistics from one VMC calculation."""

    alpha: float
    energy: float
    variance: float
    std_error: float
    acceptance_rate: float
    n_samples: int
    block_size: int
    n_blocks: int


class MetropolisSampler1D:
    """
    One-dimensional Metropolis sampler based on log-probabilities.

    Using log-probabilities avoids unnecessary underflow when the target
    probability density becomes very small.
    """

    def __init__(
        self,
        log_prob_func: ArrayFunc,
        proposal_width: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Initialise the sampler.

        Parameters
        ----------
        log_prob_func : callable
            Function returning the log of the target probability density,
            up to an additive constant.
        proposal_width : float
            Half-width of the symmetric uniform proposal distribution.
        rng : numpy.random.Generator
            Random number generator used for the Metropolis walk.
        """
        if proposal_width <= 0.0:
            raise ValueError("proposal_width must be positive.")
        self.log_prob_func = log_prob_func
        self.proposal_width = proposal_width
        self.rng = rng

    def step(
        self,
        position: float,
        log_prob_current: float,
    ) -> Tuple[float, float, bool]:
        """
        Perform a single Metropolis update.

        Returns the updated position, the corresponding log-probability,
        and a flag indicating whether the trial move was accepted.
        """
        trial = position + self.rng.uniform(-self.proposal_width, self.proposal_width)
        log_prob_trial = self.log_prob_func(trial)

        if not np.isfinite(log_prob_trial):
            return position, log_prob_current, False

        log_ratio = log_prob_trial - log_prob_current
        if log_ratio >= 0.0 or math.log(self.rng.random()) < log_ratio:
            return trial, log_prob_trial, True

        return position, log_prob_current, False
