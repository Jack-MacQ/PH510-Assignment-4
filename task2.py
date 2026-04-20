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
