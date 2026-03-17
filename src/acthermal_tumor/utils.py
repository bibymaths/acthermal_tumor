"""Utility routines for initial conditions and visualisation.

 The functions in this module generate biologically plausible
 initial conditions for the tumour growth model and provide simple
 plotting helpers.  These routines depend only on NumPy/matplotlib and
 do not require JAX.
 """

from __future__ import annotations

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from .core import State


def generate_initial_conditions(
        shape: Tuple[int, ...],
        radius_fraction: float = 0.1,
        tumour_value: float = 1.0,
        background_value: float = 0.0,
        theta_value: float = 0.0,
        sigma_value: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate simple initial conditions for testing.

    A small spherical (or circular) tumour of radius `radius_fraction` times
    the domain size is placed at the centre of the domain.  The
    temperature is initialized uniformly to `theta_value` and the
    nutrient concentration to `sigma_value`.

    Parameters
    ----------
    shape: tuple of int
        Spatial discretisation, e.g. `(Nx, Ny)` or `(Nx, Ny, Nz)`.
    radius_fraction: float
        Fraction of the smallest domain length used as the tumour radius.
    tumour_value: float
        Value of φ inside the tumour.
    background_value: float
        Value of φ outside the tumour.
    theta_value: float
        Uniform initial temperature.
    sigma_value: float
        Uniform initial nutrient concentration.

    Returns
    -------
    (phi0, theta0, sigma0): tuple of numpy.ndarray
        Arrays containing the initial values of φ, θ and σ.
    """
    # Create coordinate grids centred at zero
    coords = [np.linspace(-0.5, 0.5, n, endpoint=False) for n in shape]
    grids = np.meshgrid(*coords, indexing="ij")
    # Radial distance from centre
    r2 = np.zeros(shape)
    for g in grids:
        r2 += g ** 2
    r = np.sqrt(r2)
    # Determine radius relative to the smallest dimension
    radius = radius_fraction * 0.5
    phi0 = np.where(r <= radius, tumour_value, background_value).astype(np.float32)
    theta0 = np.full(shape, theta_value, dtype=np.float32)
    sigma0 = np.full(shape, sigma_value, dtype=np.float32)
    return phi0, theta0, sigma0


def plot_state(state: State, cmap: str = "viridis") -> None:
    """Visualise the current state of the simulation.

    For two‑dimensional problems the fields φ, θ and σ are plotted as
    separate subplots.  In higher dimensions the central slice along
    the last axis is shown.  Matplotlib must be installed to use
    this function.
    """
    phi = np.array(state.phi)
    theta = np.array(state.theta)
    sigma = np.array(state.sigma)

    ndim = phi.ndim
    if ndim > 3:
        raise ValueError("plot_state supports up to 3D data")

    # Determine slice index for higher dimensions
    if ndim == 3:
        idx = phi.shape[-1] // 2
        phi_plot = phi[..., idx]
        theta_plot = theta[..., idx]
        sigma_plot = sigma[..., idx]
    else:
        phi_plot, theta_plot, sigma_plot = phi, theta, sigma

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(phi_plot, origin="lower", cmap=cmap)
    axes[0].set_title("φ (tumour)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(theta_plot, origin="lower", cmap=cmap)
    axes[1].set_title("θ (temperature)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(sigma_plot, origin="lower", cmap=cmap)
    axes[2].set_title("σ (nutrient)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
