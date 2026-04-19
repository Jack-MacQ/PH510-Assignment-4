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
    proposal_width: flot = 1.0
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
        """Return a formated readable summary."""
        return (
            f"Energy           = {self.energy:+.8f} Hartree\n"
            f"Variance         = {self.variance:.8e} Hartree^2\n"
            f"Std. error       = {self.std_error:.8e} Hartree\n"
            f"Acceptance rate  = {self.acceptance_rate:.2%}\n"
            f"Samples          = {self.n_samples:,}\n"
            f"Block size       = {self.block_size}\n"
            f"Blocks           = {self.n_blocks}"
        )
