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


def run_vmc_1d(
    config: VMCConfig,
    log_prob_func: ArrayLikeFunc,
    local_energy_func: ArrayLikeFunc,
) -> VMCResult:
    """
    Run a one-dimensional Variational Monte Carlo calculation.

    Parameters
    ----------
    config : VMCConfig
        Configuration parameters.
    log_prob_func : callable
        Function returning log of the unnormalised target probability.
    local_energy_func : callable
        Function returning the local energy at the sampled coordinate.

    Returns
    -------
    VMCResult
        Estimated energy, variance, standard error, and run statistics.
    """
    if config.n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if config.n_equilibration < 0:
        raise ValueError("n_equilibration must be non-negative.")
    if config.decorrelation_steps < 1:
        raise ValueError("decorrelation_steps must be at least 1.")
    if config.initial_position <= 0.0:
        raise ValueError("initial_position must be positive for this radial example.")

    rng = np.random.default_rng(config.seed)
    sampler = MetropolisSampler1D(
        log_prob_func=log_prob_func,
        proposal_width=config.proposal_width,
        rng=rng,
    )

    position = config.initial_position
    log_prob_current = log_prob_func(position)
    if not np.isfinite(log_prob_current):
        raise ValueError("initial_position has non-finite log probability.")

    for _ in range(config.n_equilibration):
        position, log_prob_current, _ = sampler.step(position, log_prob_current)

    local_energies = np.empty(config.n_samples, dtype=np.float64)
    accepted_moves = 0
    total_moves = 0

    for i in range(config.n_samples):
        for _ in range(config.decorrelation_steps):
            position, log_prob_current, accepted = sampler.step(position, log_prob_current)
            accepted_moves += int(accepted)
            total_moves += 1

        local_energies[i] = local_energy_func(position)

    energy = float(np.mean(local_energies))
    variance = float(np.var(local_energies, ddof=1))
    std_error, n_blocks = blocking_standard_error(local_energies, config.block_size)
    acceptance_rate = accepted_moves / total_moves if total_moves > 0 else 0.0

    return VMCResult(
        energy=energy,
        variance=variance,
        std_error=std_error,
        acceptance_rate=acceptance_rate,
        n_samples=config.n_samples,
        block_size=config.block_size,
        n_blocks=n_blocks,
    )


def hydrogen_log_radial_probability(alpha: float) -> ArrayLikeFunc:
    """
    Build the log of the unnormalised hydrogen radial probability.

    For the 1s trial wavefunction psi_T(r, alpha) = alpha exp(-alpha r),
    the radial probability density is proportional to

        |psi_T(r)|^2 r^2 ~ exp(-2 alpha r) r^2

    because the overall multiplicative constant cancels in Metropolis
    sampling.

    Parameters
    ----------
    alpha : float
        Variational parameter, alpha > 0.

    Returns
    -------
    callable
        Function of radius returning log P(r) up to an additive constant.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    def log_prob(radius: float) -> float:
        if radius <= 0.0:
            return -np.inf
        return 2.0 * math.log(radius) - 2.0 * alpha * radius

    return log_prob


def hydrogen_local_energy(alpha: float) -> ArrayLikeFunc:
    """
    Build the hydrogen 1s local-energy function.

    Parameters
    ----------
    alpha : float
        Variational parameter, alpha > 0.

    Returns
    -------
    callable
        Function of radius returning the local energy in Hartree.
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    def local_energy(radius: float) -> float:
        return -1.0 / radius - 0.5 * alpha * (alpha - 2.0 / radius)

    return local_energy


def hydrogen_demo(alpha: float = 1.0) -> VMCResult:
    """
    Run a demonstration VMC calculation for hydrogen 1s.

    Parameters
    ----------
    alpha : float, optional
        Variational parameter.

    Returns
    -------
    VMCResult
        VMC result for the chosen alpha.
    """
    config = VMCConfig(
        n_samples=200_000,
        n_equilibration=10_000,
        decorrelation_steps=5,
        proposal_width=1.0 / alpha,
        initial_position=1.0 / alpha,
        seed=42,
        block_size=200,
    )

    return run_vmc_1d(
        config=config,
        log_prob_func=hydrogen_log_radial_probability(alpha),
        local_energy_func=hydrogen_local_energy(alpha),
    )


def main() -> None:
    """Run the hydrogen Task 1 demonstration."""
    alpha = 1.0
    result = hydrogen_demo(alpha)

    print("=" * 60)
    print("PH510 Assignment 5 - Task 1")
    print("Variational Monte Carlo engine")
    print("Hydrogen 1s demonstration")
    print("=" * 60)
    print(f"alpha            = {alpha:.6f}")
    print(result)
    print("\nReference values at alpha = 1:")
    print("Exact energy     = -0.50000000 Hartree")
    print("Exact variance   = 0.00000000 Hartree^2")
    print(f"|E + 0.5|        = {abs(result.energy + 0.5):.8e}")


if __name__ == "__main__":
    main()
