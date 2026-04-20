"""
PH510 Assignment 4 - Task 3
Variational Monte Carlo study of two hard-sphere bosons in a 2D harmonic trap.

This script applies the Variational Monte Carlo (VMC) method to estimate
the ground-state energy of two identical bosons confined in a
two-dimensional isotropic harmonic oscillator potential.

Copyright (c) 2026 Jack MacQuarrie

This code is released under the MIT License. See the LICENSE file in the
repository for details.

Python Version: 3.9.21
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data containers for run parameters and measured observables
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BosonsConfig:
    """
    Numerical parameters controlling a single VMC run.

    The defaults are chosen to give a reasonable compromise between
    statistical quality and runtime for the two-boson problem.
    """

    n_samples: int = 50_000
    n_equilibration: int = 5_000
    decorrelation_steps: int = 5
    proposal_width: float = 0.8
    seed: int = 42
    block_size: int = 100
    hard_sphere_diameter: float = 0.0043


@dataclass(frozen=True)
class BosonsResult:
    """
    Container for the output of a single VMC calculation.

    This stores the variational parameters together with the estimated
    energy, variance, statistical uncertainty, and Metropolis acceptance
    rate.
    """

    alpha: float
    beta: float
    energy: float
    variance: float
    std_error: float
    acceptance_rate: float
    n_samples: int

