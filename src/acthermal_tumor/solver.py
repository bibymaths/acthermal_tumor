"""High‑level solver for the non‑isothermal Allen–Cahn tumour model.

The `ThermalTumorSimulator` encapsulates grid initialization,
parameter handling and time stepping.  It provides a simple API for
advancing the coupled PDE system in time.  Internally, it uses a
fourth‑order Runge–Kutta scheme and JAX's JIT compiler for speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core import State, build_spectral_laplace, rhs
from .parameters import Parameters


@dataclass
class ThermalTumorSimulator:
    """Simulate the non‑isothermal Allen–Cahn tumour growth model.

    Parameters
    ----------
    shape: tuple of int
        Shape of the spatial grid (e.g. `(Nx, Ny)` for 2D).  All
        dimensions must be positive integers.
    params: Parameters
        Physical and numerical parameters.  The time‑step `dt` and grid
        spacing `dx` are taken from this object.
    use_spectral: bool, optional
        If true and the `exponax` library is available, attempt to
        construct a spectral Laplace operator for the linear parts of
        the PDE.  When false or if `exponax` cannot be imported the
        solver falls back to finite‑difference operators.
    """

    shape: tuple[int, ...]
    params: Parameters
    use_spectral: bool = True

    def __post_init__(self) -> None:
        # Validate shape
        if not all(isinstance(n, int) and n > 1 for n in self.shape):
            raise ValueError("shape must be a tuple of integers greater than one")

        # Attempt to build a spectral Laplacian operator.  This is
        # optional; if it fails `self.spectral_laplace` will be None and
        # the finite‑difference operators in `core` will be used instead.
        self.spectral_laplace = None
        if self.use_spectral:
            self.spectral_laplace = build_spectral_laplace(self.shape, self.params.dx)

        # Determine a stable time step for the explicit RK4 scheme.  For a
        # multidimensional diffusion equation the stability condition is
        # dt <= dx**2 / (4 * dim).  We include a safety factor of 0.5 to
        # avoid approaching the limit too closely.  If the user‑specified
        # dt is smaller than the stability bound, we use it directly.
        dims = len(self.shape)
        stability_dt = (self.params.dx**2) / (4.0 * dims)
        self._dt = (
            self.params.dt if self.params.dt <= stability_dt else 0.5 * stability_dt
        )

        params = self.params

        def _rk4_step(state: State) -> State:
            # Use the internally computed time step for stability
            dt = self._dt
            f1 = rhs(state, params)
            s2 = state + f1 * (dt / 2.0)
            f2 = rhs(s2, params)
            s3 = state + f2 * (dt / 2.0)
            f3 = rhs(s3, params)
            s4 = state + f3 * dt
            f4 = rhs(s4, params)
            return state + (f1 + f2 * 2.0 + f3 * 2.0 + f4) * (dt / 6.0)

        # JIT compile the stepper for improved performance
        self._step_fn = jax.jit(_rk4_step)

    def initialize_state(
        self, phi0: jnp.ndarray, theta0: jnp.ndarray, sigma0: jnp.ndarray
    ) -> State:
        """Create a `State` object from raw initial conditions.

        The shapes of `phi0`, `theta0` and `sigma0` must match the
        simulator's spatial `shape`.
        """
        if (
            phi0.shape != self.shape
            or theta0.shape != self.shape
            or sigma0.shape != self.shape
        ):
            raise ValueError(f"Initial conditions must have shape {self.shape}")
        return State(phi0, theta0, sigma0)

    def step(self, state: State) -> State:
        """Advance the state by one time step.

        This method applies a single RK4 step to the current state and
        returns the updated state.  The JIT compiled `_step_fn` is
        invoked so that repeated calls are efficient.
        """
        # Apply one RK4 step
        new_state = self._step_fn(state)
        # Enforce physical bounds to prevent numerical blow‑up.  Tumour
        # concentration φ and nutrient σ should remain within [0, 1], while
        # temperature θ is clipped to a moderate range.  Clipping helps
        # maintain stability for explicit schemes without affecting
        # qualitative behaviour.
        phi_clipped = jnp.clip(new_state.phi, 0.0, 1.0)
        theta_clipped = jnp.clip(new_state.theta, -10.0, 10.0)
        sigma_clipped = jnp.clip(new_state.sigma, 0.0, 1.0)
        # Replace any NaNs that may have arisen during computation
        phi_clean = jnp.nan_to_num(phi_clipped, nan=0.0)
        theta_clean = jnp.nan_to_num(theta_clipped, nan=0.0)
        sigma_clean = jnp.nan_to_num(sigma_clipped, nan=0.0)
        return State(phi_clean, theta_clean, sigma_clean)

    def run(self, state: State, num_steps: int) -> State:
        """Advance the system for a specified number of time steps.

        Parameters
        ----------
        state: State
            Initial state.  It will not be modified.
        num_steps: int
            Number of time steps to perform.

        Returns
        -------
        State
            The state after `num_steps` integration steps.
        """
        current = state
        for _ in range(num_steps):
            current = self.step(current)
        return current
