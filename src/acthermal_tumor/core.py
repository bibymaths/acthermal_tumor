"""Core constitutive functions and discrete differential operators.

This module implements the mathematical building blocks of the
non‑isothermal Allen–Cahn tumour growth model.  It defines a
`State` container for the three dynamical variables, constitutive
functions such as the double‑well potential and its derivative, and
discrete differential operators that enforce homogeneous Neumann
boundary conditions on a regular grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp

from .parameters import Parameters

try:
    # Attempt to import exponax.  It is optional and used to build
    # spectral Laplace operators when available.  If the import fails,
    # the solver falls back to finite differences.
    import exponax  # type: ignore
except Exception:
    exponax = None  # type: ignore


@jax.tree_util.register_pytree_node_class
@dataclass
class State:
    """Container class for the three dynamical variables.

    Attributes
    ----------
    phi: jnp.ndarray
        Tumour cell concentration field.
    theta: jnp.ndarray
        Temperature field.
    sigma: jnp.ndarray
        Nutrient concentration field.
    """

    phi: jnp.ndarray
    theta: jnp.ndarray
    sigma: jnp.ndarray

    def __add__(self, other: 'State') -> 'State':
        return State(self.phi + other.phi,
                     self.theta + other.theta,
                     self.sigma + other.sigma)

    def __mul__(self, scalar: float) -> 'State':
        return State(self.phi * scalar,
                     self.theta * scalar,
                     self.sigma * scalar)

    __rmul__ = __mul__

    # PyTree methods for JAX: treat the State as a container of arrays.
    # These methods allow JAX transformations (jit, vmap, etc.) to
    # operate transparently on State instances by flattening them into
    # their constituent array leaves.
    def tree_flatten(self):
        # Children are the array fields; no auxiliary static data
        return (self.phi, self.theta, self.sigma), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        phi, theta, sigma = children
        return cls(phi, theta, sigma)


def double_well_potential(phi: jnp.ndarray) -> jnp.ndarray:
    """Compute the double‑well potential F(φ) = φ²(1−φ)².

    The potential has minima at φ=0 and φ=1 and penalises
    intermediate values.  It appears in the Allen–Cahn equation.

    Parameters
    ----------
    phi: array_like
        Tumour cell concentration field.

    Returns
    -------
    array_like
        Value of the potential at each point.
    """
    return phi ** 2 * (1.0 - phi) ** 2


def double_well_prime(phi: jnp.ndarray) -> jnp.ndarray:
    """Derivative of the double‑well potential.

    Computes F′(φ) = 4 φ³ − 6 φ² + 2 φ.
    """
    return 4.0 * phi ** 3 - 6.0 * phi ** 2 + 2.0 * phi


def activation_function(phi: jnp.ndarray) -> jnp.ndarray:
    """Smooth activation function h(φ).

    A smooth approximation of `max(0, φ)`.  The model requires
    `h` to be monotone increasing, bounded and satisfy h(0)=0.
    A shifted softplus is used: `h(φ) = softplus(φ) - log(2)` so that h(0)=0.
    """
    return jax.nn.softplus(phi) - jnp.log(2.0)


def kappa(theta: jnp.ndarray, q: int = 2) -> jnp.ndarray:
    """Heat conductivity κ(θ)【901098191607575†L470-L474】.

    Computes κ(θ) = 1 + |θ|^q.

    Parameters
    ----------
    theta: array_like
        Temperature field.
    q: int
        Exponent controlling the nonlinearity of the conductivity.

    Returns
    -------
    array_like
        Pointwise conductivity.
    """
    return 1.0 + jnp.abs(theta) ** q


def _shift_forward_neumann(u: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Shift array forward along a given axis with Neumann reflection.

    For an interior index *i* the function returns u[i+1].  At the upper
    boundary a mirrored ghost cell equal to u[-2] is used so that the
    discrete normal derivative vanishes.

    Parameters
    ----------
    u: array_like
        Input array.
    axis: int
        Axis along which to shift.

    Returns
    -------
    array_like
        Shifted array.
    """
    n = u.shape[axis]
    idxs = jnp.arange(n)
    forward_idx = jnp.clip(idxs + 1, 0, n - 1)
    return jnp.take(u, forward_idx, axis=axis)


def _shift_backward_neumann(u: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Shift array backward along a given axis with Neumann reflection.

    For an interior index *i* the function returns u[i−1].  At the lower
    boundary a mirrored ghost cell equal to u[1] is used so that the
    discrete normal derivative vanishes.
    """
    n = u.shape[axis]
    idxs = jnp.arange(n)
    backward_idx = jnp.clip(idxs - 1, 0, n - 1)
    return jnp.take(u, backward_idx, axis=axis)


def laplacian_neumann(u: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Compute the discrete Laplacian of `u` with Neumann boundary conditions.

    A central difference stencil is used in each spatial dimension.  At
    the boundaries the ghost values are mirrored to enforce zero normal
    derivative.  This implementation works for arbitrary dimensionality
    and is JIT‑compatible.
    """
    lap = jnp.zeros_like(u)
    for axis in range(u.ndim):
        fwd = _shift_forward_neumann(u, axis)
        bwd = _shift_backward_neumann(u, axis)
        lap = lap + (fwd - 2.0 * u + bwd)
    return lap / (dx ** 2)


def gradient_neumann(u: jnp.ndarray, dx: float) -> List[jnp.ndarray]:
    """Compute the gradient of a scalar field with Neumann boundaries.

    Central differences are used in each spatial dimension and ghost
    values are mirrored to satisfy the zero normal derivative condition.
    """
    grads: List[jnp.ndarray] = []
    for axis in range(u.ndim):
        fwd = _shift_forward_neumann(u, axis)
        bwd = _shift_backward_neumann(u, axis)
        grads.append((fwd - bwd) / (2.0 * dx))
    return grads


def divergence_neumann(vector_components: List[jnp.ndarray], dx: float) -> jnp.ndarray:
    """Compute the divergence of a vector field with Neumann boundaries.

    Each component of the vector field must have the same shape.  A
    central difference approximation is used for the derivative.
    """
    div = jnp.zeros_like(vector_components[0])
    for axis, comp in enumerate(vector_components):
        fwd = _shift_forward_neumann(comp, axis)
        bwd = _shift_backward_neumann(comp, axis)
        div = div + (fwd - bwd) / (2.0 * dx)
    return div


def rhs(state: State, params: Parameters) -> State:
    """Compute the right‑hand side of the tumour growth system.

    Returns a `State` containing time derivatives `(φ_t, θ_t, σ_t)`
    evaluated at the current state.  The derivative of φ is computed
    first because it appears in the temperature equation.
    """
    phi = state.phi
    theta = state.theta
    sigma = state.sigma
    dx = params.dx

    # Laplacians
    lap_phi = laplacian_neumann(phi, dx)
    lap_sigma = laplacian_neumann(sigma, dx)

    # Allen–Cahn equation
    phi_t = lap_phi - double_well_prime(phi) + theta + (params.P * sigma - params.A) * activation_function(phi)

    # Temperature equation: ∂tθ = div(κ(θ)∇θ) + φ_t² − θ φ_t
    theta_grad = gradient_neumann(theta, dx)
    conductivity = kappa(theta, q=params.q)
    flux_components = [conductivity * g for g in theta_grad]
    div_flux = divergence_neumann(flux_components, dx)
    theta_t = div_flux + phi_t ** 2 - theta * phi_t

    # Nutrient equation
    sigma_t = lap_sigma - params.C * sigma * activation_function(phi) + params.B * (params.sigma_B - sigma)

    return State(phi_t, theta_t, sigma_t)


def build_spectral_laplace(shape: Tuple[int, ...], dx: float) -> Optional[object]:
    """Create a spectral Laplace operator using exponax, if available.

    Exponax operates on periodic domains.  Neumann conditions can be
    approximated by mirroring the domain before applying the spectral
    operator.  If exponax is not available, return `None`.
    """
    if exponax is None:
        return None
    lengths = tuple(n * dx for n in shape)
    try:
        laplace = exponax.LinearOperators.laplace_operator(lengths=lengths, dim=len(shape))
    except Exception:
        laplace = None
    return laplace
