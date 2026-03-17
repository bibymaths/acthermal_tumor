"""Integration tests for the thermal tumour growth solver.

These tests verify that the solver can advance the coupled PDE
system without producing NaNs under mild conditions.  Because
`exponax` may not be installed in the test environment, the test
gracefully falls back to the finite‑difference implementation by
disabling spectral operators.
"""

import pytest

# Attempt to import jax.numpy; skip tests if JAX is unavailable
jnp = pytest.importorskip("jax.numpy", reason="jax is required to run the solver tests")

import os
import sys

# Add the project's src directory to the Python path so that
# `acthermal_tumor` can be imported without installation.
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, "src"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from acthermal_tumor.parameters import Parameters
from acthermal_tumor.solver import ThermalTumorSimulator
from acthermal_tumor.utils import generate_initial_conditions


def test_solver_runs_without_nan() -> None:
    """Run the simulator for a few steps and ensure no NaNs arise."""
    shape = (32, 32)
    params = Parameters(P=1.0, A=0.5, C=1.0, B=0.2, sigma_B=1.0, q=2, dt=1e-3, dx=1.0 / 32)
    phi0, theta0, sigma0 = generate_initial_conditions(shape, radius_fraction=0.1)

    sim = ThermalTumorSimulator(shape=shape, params=params, use_spectral=False)
    state = sim.initialize_state(jnp.array(phi0), jnp.array(theta0), jnp.array(sigma0))

    # Run a few steps
    for _ in range(5):
        state = sim.step(state)

    # Assert that no NaN values are present
    assert not jnp.isnan(state.phi).any(), "NaNs detected in phi after stepping"
    assert not jnp.isnan(state.theta).any(), "NaNs detected in theta after stepping"
    assert not jnp.isnan(state.sigma).any(), "NaNs detected in sigma after stepping"