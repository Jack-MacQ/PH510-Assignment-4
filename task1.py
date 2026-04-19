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
