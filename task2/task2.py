"""
PH510 Assignment 5 - Task 2
Variational Monte Carlo study of the hydrogen 1s ground state.

This script applies the Variational Monte Carlo (VMC) method to the
hydrogen 1s radial problem using a trial wavefunction with variational
parameter alpha. For each chosen alpha value, a Metropolis random walk
is used to sample the radial probability distribution, from which the
local energy, energy expectation value, variance, and statistical
uncertainty are estimated.

The code performs a scan over positive alpha values, identifies the
parameter giving the lowest energy and variance, writes a formatted
summary of the results to file, and produces plots of energy and
variance as functions of alpha.

For the exact hydrogen ground state, the optimal value is alpha = 1,
for which the ground-state energy is -0.5 Hartree and the local-energy
variance is zero.

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


# pylint: disable=too-many-instance-attributes
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


# pylint: disable=too-few-public-methods
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


# pylint: disable=too-many-locals
def run_vmc(config: VMCConfig, alpha: float) -> VMCResult:
    """
    Run one VMC calculation for a given alpha.

    The Markov chain is first equilibrated, then local-energy samples are
    collected during the production phase. A small number of additional
    Metropolis steps can be inserted between recorded samples to reduce
    autocorrelation.

    Parameters
    ----------
    config : VMCConfig
        Numerical settings for the run.
    alpha : float
        Variational parameter for the hydrogen trial wavefunction.

    Returns
    -------
    VMCResult
        Energy, variance, standard error, and basic sampling statistics.
    """
    if config.n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if config.n_equilibration < 0:
        raise ValueError("n_equilibration must be non-negative.")
    if config.decorrelation_steps < 1:
        raise ValueError("decorrelation_steps must be at least 1.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    rng = np.random.default_rng(config.seed)
    log_prob_func = hydrogen_log_radial_probability(alpha)
    local_energy_func = hydrogen_local_energy(alpha)

    sampler = MetropolisSampler1D(
        log_prob_func=log_prob_func,
        proposal_width=config.proposal_width,
        rng=rng,
    )

    position = config.initial_position
    log_prob_current = log_prob_func(position)

    if not np.isfinite(log_prob_current):
        raise ValueError("Initial position has non-finite log probability.")

    # Equilibration stage: allow the Markov chain to relax before sampling.
    for _ in range(config.n_equilibration):
        position, log_prob_current, _ = sampler.step(position, log_prob_current)

    local_energies = np.empty(config.n_samples, dtype=np.float64)
    accepted_moves = 0
    total_moves = 0

    # Production stage: record one local-energy sample after each set of
    # decorrelation moves.
    for i in range(config.n_samples):
        for _ in range(config.decorrelation_steps):
            position, log_prob_current, accepted = sampler.step(
                position, log_prob_current
            )
            accepted_moves += int(accepted)
            total_moves += 1

        local_energies[i] = local_energy_func(position)

    energy = float(np.mean(local_energies))
    variance = float(np.var(local_energies, ddof=1))
    std_error, n_blocks = blocking_standard_error(local_energies, config.block_size)
    acceptance_rate = accepted_moves / total_moves if total_moves > 0 else 0.0

    return VMCResult(
        alpha=alpha,
        energy=energy,
        variance=variance,
        std_error=std_error,
        acceptance_rate=acceptance_rate,
        n_samples=config.n_samples,
        block_size=config.block_size,
        n_blocks=n_blocks,
    )


def scan_alpha(alpha_values: np.ndarray, base_seed: int = 42) -> list[VMCResult]:
    """
    Run the VMC calculation over a range of alpha values.

    A different random seed is used for each alpha so that neighbouring
    points in the scan do not reuse the same random sequence.

    Parameters
    ----------
    alpha_values : numpy.ndarray
        Array of alpha values to test.
    base_seed : int, optional
        Starting seed for the scan.

    Returns
    -------
    list[VMCResult]
        Results for each alpha value in the scan.
    """
    results: list[VMCResult] = []

    for i, alpha in enumerate(alpha_values):
        config = VMCConfig(
            n_samples=100_000,
            n_equilibration=10_000,
            decorrelation_steps=5,
            proposal_width=1.75 / alpha,
            initial_position=1.0 / alpha,
            seed=base_seed + i,
            block_size=100,
        )
        results.append(run_vmc(config, alpha))

    return results


def print_summary(results: list[VMCResult]) -> None:
    """
    Print a formatted summary table for the alpha scan.

    The table lists the energy, estimated uncertainty, variance, and
    acceptance rate for each alpha, then highlights the best-energy and
    lowest-variance cases.
    """
    print("=" * 86)
    print("PH510 Assignment 5 - Task 2")
    print("Hydrogen 1s ground state from Variational Monte Carlo")
    print("=" * 86)
    print(
        f"{'alpha':>8}  {'energy / Ha':>16}  {'std err':>12}  "
        f"{'variance / Ha^2':>18}  {'acceptance':>11}"
    )
    print("-" * 86)

    for res in results:
        print(
            f"{res.alpha:8.4f}  {res.energy:16.8f}  {res.std_error:12.4e}  "
            f"{res.variance:18.8e}  {res.acceptance_rate:10.2%}"
        )

    best_energy = min(results, key=lambda item: item.energy)
    best_variance = min(results, key=lambda item: item.variance)

    print("-" * 86)
    print("Best energy:")
    print(
        f"  alpha = {best_energy.alpha:.2f}, "
        f"E = {best_energy.energy:.2f} +/- {best_energy.std_error:.2e} Ha"
    )
    print("Lowest variance:")
    print(
        f"  alpha = {best_variance.alpha:.2f}, "
        f"Var = {best_variance.variance:.2e} Ha^2"
    )
    print("Exact hydrogen ground-state energy: -0.50 Ha")


def save_results_txt(
    results: list[VMCResult],
    filename: str = "task2_results.txt",
) -> None:
    """
    Save the formatted alpha-scan summary to a text file.

    This writes the same information shown on screen to a plain-text file,
    which can be used as a record of the scan or as a source for tables
    in the report.
    """
    with open(filename, "w", encoding="utf-8") as file_out:
        file_out.write("=" * 86 + "\n")
        file_out.write("PH510 Assignment 5 - Task 2\n")
        file_out.write("Hydrogen 1s ground state from Variational Monte Carlo\n")
        file_out.write("=" * 86 + "\n")
        file_out.write(
            f"{'alpha':>8}  {'energy / Ha':>16}  {'std err':>12}  "
            f"{'variance / Ha^2':>18}  {'acceptance':>11}\n"
        )
        file_out.write("-" * 86 + "\n")

        for res in results:
            file_out.write(
                f"{res.alpha:8.4f}  {res.energy:16.8f}  {res.std_error:12.4e}  "
                f"{res.variance:18.8e}  {res.acceptance_rate:10.2%}\n"
            )

        best_energy = min(results, key=lambda item: item.energy)
        best_variance = min(results, key=lambda item: item.variance)

        file_out.write("-" * 86 + "\n")
        file_out.write("Best energy:\n")
        file_out.write(
            f"  alpha = {best_energy.alpha:.2f}, "
            f"E = {best_energy.energy:.2f} +/- {best_energy.std_error:.2e} Ha\n"
        )
        file_out.write("Lowest variance:\n")
        file_out.write(
            f"  alpha = {best_variance.alpha:.2f}, "
            f"Var = {best_variance.variance:.2e} Ha^2\n"
        )
        file_out.write("Exact hydrogen ground-state energy: -0.50 Ha\n")


def plot_results(results: list[VMCResult]) -> None:
    """
    Plot the energy and variance as functions of alpha.

    Two figures are produced: one showing the energy with error bars and
    the exact value for comparison, and one showing the variance across
    the scan.
    """
    alpha_vals = np.array([res.alpha for res in results], dtype=float)
    energies = np.array([res.energy for res in results], dtype=float)
    errors = np.array([res.std_error for res in results], dtype=float)
    variances = np.array([res.variance for res in results], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.errorbar(alpha_vals, energies, yerr=errors, fmt="o-", capsize=3)
    plt.axhline(-0.5, linestyle="--", label="Exact energy")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Energy / Hartree")
    plt.title("Hydrogen 1s VMC energy")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task2_energy_vs_alpha.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_vals, variances, "o-")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"Variance / Hartree$^2$")
    plt.title("Hydrogen 1s VMC variance")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig("task2_variance_vs_alpha.png", dpi=300)
    plt.show()


def main() -> None:
    """
    Run the hydrogen alpha scan and generate the summary output.

    The scan range can be adjusted here if a finer search around the
    minimum is required.
    """
    alpha_values = np.linspace(0.5, 1.5, 21)
    results = scan_alpha(alpha_values)
    print_summary(results)
    save_results_txt(results)
    plot_results(results)


if __name__ == "__main__":
    main()
