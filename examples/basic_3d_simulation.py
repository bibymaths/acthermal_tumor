"""
Basic 3D example for acthermal_tumor.

This script:
1. Builds a 3D simulation domain
2. Initializes a small tumor in the center
3. Runs the non-isothermal Allen-Cahn tumor model
4. Visualizes the final state

Run from the project root with, for example:
    python examples/basic_3d_simulation.py
"""

import jax.numpy as jnp

from acthermal_tumor.parameters import Parameters
from acthermal_tumor.solver import ThermalTumorSimulator
from acthermal_tumor.utils import generate_initial_conditions, plot_state


def main() -> None:
    # Set up a 3D simulation domain
    shape = (32, 32, 32)

    params = Parameters(
        P=1.5,
        A=1.0,
        C=1.2,
        B=0.3,
        sigma_B=0.8,
        q=2,
        dt=5e-4,
        dx=1.0 / 32,
    )

    # Generate initial conditions:
    # small spherical tumor, uniform temperature, uniform nutrient
    phi0, theta0, sigma0 = generate_initial_conditions(
        shape=shape,
        radius_fraction=0.08,
    )

    # Build simulator
    sim = ThermalTumorSimulator(shape=shape, params=params)

    # Convert initial conditions to JAX arrays and initialize state
    state = sim.initialize_state(
        jnp.array(phi0),
        jnp.array(theta0),
        jnp.array(sigma0),
    )

    # Integrate for 200 steps
    state = sim.run(state, num_steps=10000)

    # Visualize the final state
    # For 3D data, plot_state shows the central slice automatically
    plot_state(state)


if __name__ == "__main__":
    main()
