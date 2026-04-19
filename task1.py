"""
PH510 Assignment 5 - Task 1
Variational Monte Carlo (VMC) engine implementation.

This script implements a general Variational Monte Carlo framework for
estimating the energy expectation value and variance of a trial
wavefunction using the local energy formalism.

A Metropolis algorithm is used to sample configurations from a target
probability distribution (known up to a normalisation constant), and
statistical estimates of the energy and its variance are obtained from
the sampled local energies.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import math
import numpy as np


ArrayLikeFunc = Callable[[float], float]


@dataclass(frozen=True)
class VMCConfig:
    """Configuration parameters for a VMC run."""

    n_samples: int = 200_000
    n_equilibration: int = 10_000
    decorrelation_steps: int = 5
    proposal_width: float = 1.0
    initial_position: float = 1.0
    seed: int = 42
    block_size: int = 200


@dataclass(frozen=True)
class VMCResult:
    """Container holding VMC observables and run statistics."""

    energy: float
    variance: float
    std_error: float
    acceptance_rate: float
    n_samples: int
    block_size: int
    n_blocks: int

    def __str__(self) -> str:
        """Return a formatted readable summary."""
        return (
            f"Energy           = {self.energy:+.8f} Hartree\n"
            f"Variance         = {self.variance:.8e} Hartree^2\n"
            f"Std. error       = {self.std_error:.8e} Hartree\n"
            f"Acceptance rate  = {self.acceptance_rate:.2%}\n"
            f"Samples          = {self.n_samples:,}\n"
            f"Block size       = {self.block_size}\n"
            f"Blocks           = {self.n_blocks}"
        )


def blocking_standard_error(values: np.ndarray, block_size: int) -> Tuple[float, int]:
    """
    Estimate the standard error from block averages.

    This helps reduce the bias caused by autocorrelation in Metropolis data.
    The returned value is the standard error of the mean estimated from
    approximately independent block averages.

    Parameters
    ----------
    values : numpy.ndarray
        Sampled observable values.
    block_size : int
        Number of consecutive samples per block.

    Returns
    -------
    tuple[int, float]
        Standard error estimate and number of complete blocks used.

    Raises
    ------
    ValueError
        If there are not enough samples to form at least two complete blocks.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    n_blocks = values.size // block_size
    if n_blocks < 2:
        raise ValueError(
            "Need at least two complete blocks to estimate blocked standard error."
        )

    trimmed = values[: n_blocks * block_size]
    block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    std_error = float(np.sqrt(np.var(block_means, ddof=1) / n_blocks))
    return std_error, n_blocks


class MetropolisSampler1D:
    """
    One-dimensional Metropolis sampler.

    The sampler targets a probability density known up to an overall
    normalisation constant by using log-probabilities.
    """

    def __init__(
        self,
        log_prob_func: ArrayLikeFunc,
        proposal_width: float,
        rng: np.random.Generator,
    ) -> None:
        """
        Initialise the sampler.

        Parameters
        ----------
        log_prob_func : callable
            Function returning log of the unnormalised target probability.
        proposal_width : float
            Half-width of the symmetric uniform proposal distribution.
        rng : numpy.random.Generator
            Random number generator.
        """
        if proposal_width <= 0.0:
            raise ValueError("proposal_width must be positive.")

        self.log_prob_func = log_prob_func
        self.proposal_width = proposal_width
        self.rng = rng

    def step(self, position: float, log_prob_current: float) -> Tuple[float, float, bool]:
        """
        Perform one Metropolis update.

        A trial point is drawn from a symmetric uniform proposal,
        so the acceptance probability is

            min(1, exp(log P_trial - log P_current))

        Parameters
        ----------
        position : float
            Current position.
        log_prob_current : float
            Current log-probability.

        Returns
        -------
        tuple
            New position, new log-probability, and acceptance flag.
        """
        trial = position + self.rng.uniform(-self.proposal_width, self.proposal_width)
        log_prob_trial = self.log_prob_func(trial)

        if not np.isfinite(log_prob_trial):
            return position, log_prob_current, False

        log_ratio = log_prob_trial - log_prob_current
        if log_ratio >= 0.0 or math.log(self.rng.random()) < log_ratio:
            return trial, log_prob_trial, True

        return position, log_prob_current, False
