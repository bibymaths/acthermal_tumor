"""
2D example comparing two different parameter regimes:
1. Aggressive growth (High Proliferation, High Nutrient Supply)
2. Starvation/Decay (High Apoptosis, Low Nutrient Supply)
"""

import jax.numpy as jnp

from acthermal_tumor.parameters import Parameters
from acthermal_tumor.solver import ThermalTumorSimulator
from acthermal_tumor.utils import generate_initial_conditions, plot_state


def run_scenario(name: str, params: Parameters, steps: int = 1500) -> None:
    print(f"--- Running Scenario: {name} ---")
    shape = (64, 64)
    phi0, theta0, sigma0 = generate_initial_conditions(shape, radius_fraction=0.15)

    sim = ThermalTumorSimulator(shape=shape, params=params)
    state = sim.initialize_state(jnp.array(phi0), jnp.array(theta0), jnp.array(sigma0))

    state = sim.run(state, num_steps=steps)
    print(f"Finished {name}. Plotting...")
    plot_state(state)


def main() -> None:
    # Scenario 1: Aggressive Growth
    aggressive_params = Parameters(
        P=3.0, A=0.2, C=0.5, B=1.0, sigma_B=1.0, dt=1e-3, dx=1.0 / 64
    )

    # Scenario 2: Starvation and Decay
    decay_params = Parameters(
        P=0.5, A=2.0, C=1.5, B=0.1, sigma_B=0.2, dt=1e-3, dx=1.0 / 64
    )

    run_scenario("Aggressive Growth", aggressive_params)
    run_scenario("Starvation and Apoptosis", decay_params)


if __name__ == "__main__":
    main()