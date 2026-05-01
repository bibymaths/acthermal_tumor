"""
2D simulation tracking the total tumor mass over time.

This script demonstrates how to extract quantitative metrics
(like total tumor cell concentration) from the State object
as the simulation progresses.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from acthermal_tumor.parameters import Parameters
from acthermal_tumor.solver import ThermalTumorSimulator
from acthermal_tumor.utils import generate_initial_conditions, plot_state


def main() -> None:
    shape = (64, 64)
    # Favorable growth parameters
    params = Parameters(
        P=2.5, A=0.5, C=1.0, B=0.8, sigma_B=1.0, q=2, dt=1e-3, dx=1.0 / 64
    )

    phi0, theta0, sigma0 = generate_initial_conditions(shape, radius_fraction=0.05)
    sim = ThermalTumorSimulator(shape=shape, params=params)

    state = sim.initialize_state(jnp.array(phi0), jnp.array(theta0), jnp.array(sigma0))

    num_steps = 3000
    record_interval = 50
    mass_history = []
    time_history = []

    print("Running simulation...")
    for step in range(num_steps):
        state = sim.step(state)

        # Record the total tumor mass (sum of phi) periodically
        if step % record_interval == 0:
            current_mass = float(jnp.sum(state.phi))
            mass_history.append(current_mass)
            time_history.append(step * params.dt)

    # Plot the final spatial state
    plot_state(state)

    # Plot the mass growth curve
    plt.figure(figsize=(8, 4))
    plt.plot(
        time_history,
        mass_history,
        label=r"Total Tumor Mass ($\int \phi$)",
        color="firebrick",
        lw=2,
    )
    plt.xlabel("Time")
    plt.ylabel("Mass")
    plt.title("Tumor Growth Dynamics Over Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
