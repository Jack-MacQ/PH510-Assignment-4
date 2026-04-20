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


def blocking_standard_error(values: np.ndarray, block_size: int) -> Tuple[float, int]:
    """
    Estimate the standard error of the mean using block averaging.

    Consecutive Metropolis samples are correlated, so the naive estimate
    sigma / sqrt(N) can understate the true uncertainty. Block averaging
    partially corrects for this by grouping neighbouring samples into
    larger, approximately independent blocks.

    Parameters
    ----------
    values : numpy.ndarray
        Array of sampled values.
    block_size : int
        Number of consecutive samples in each block.

    Returns
    -------
    Tuple[float, int]
        Estimated standard error and the number of complete blocks used.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    n_blocks = values.size // block_size
    if n_blocks < 2:
        raise ValueError("Need at least two complete blocks.")

    trimmed = values[: n_blocks * block_size]
    block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    std_error = float(np.sqrt(np.var(block_means, ddof=1) / n_blocks))
    return std_error, n_blocks


def hydrogen_log_radial_probability(alpha: float) -> ArrayFunc:
    """
    Return the log of the unnormalised radial probability density.

    For the hydrogen 1s trial wavefunction, the radial probability
    density is proportional to |psi(r)|^2 r^2. The overall normalisation
    constant is omitted, since it cancels in the Metropolis acceptance
    ratio.

    Parameters
    ----------
    alpha : float
        Positive variational parameter.

    Returns
    -------
    callable
        Function returning log P(r) up to an additive constant.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    def log_prob(radius: float) -> float:
        """Evaluate the log radial probability density at radius r."""
        if radius <= 0.0:
            return -np.inf
        return 2.0 * math.log(radius) - 2.0 * alpha * radius

    return log_prob


def hydrogen_local_energy(alpha: float) -> ArrayFunc:
    """
    Return the analytical local-energy function for the hydrogen trial state.

    Parameters
    ----------
    alpha : float
        Positive variational parameter.

    Returns
    -------
    callable
        Function returning the local energy in Hartree.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    def local_energy(radius: float) -> float:
        """Evaluate the local energy at radius r."""
        return -1.0 / radius - 0.5 * alpha * (alpha - 2.0 / radius)

    return local_energy
