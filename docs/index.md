# acthermal\_tumor Documentation

Welcome to the documentation for **acthermal\_tumor**, a Python
package for simulating the non‑isothermal Allen–Cahn tumour growth
model.  This document provides a brief overview of the theory
underlying the model, installation and usage instructions, and some
examples to get you started.

## Theory overview

The non‑isothermal Allen–Cahn model couples tumour cell concentration
(`φ`), temperature (`θ`) and nutrient concentration (`σ`).  After
non‑dimensionalisation and with homogeneous Neumann boundary
conditions the system can be written as

```text
∂t φ = Δφ − F′(φ) + θ + (P σ − A) h(φ)
∂t θ = ∇·(κ(θ) ∇θ) + (∂t φ)² − θ ∂t φ
∂t σ = Δσ − C σ h(φ) + B (σ_B − σ)
```

Here the Laplacian `Δ` and the gradient/divergence operators act in
space.  The derivative of the double‑well potential is
`F′(φ) = 4 φ³ − 6 φ² + 2 φ`, the activation
function `h` is monotone increasing with `h(0) = 0` and
the conductivity is `κ(θ) = 1 + |θ|^q`.  Positive
constants `P`, `A`, `C`, `B` and `σ_B` control proliferation,
apoptosis, nutrient consumption, nutrient supply and vascular nutrient
concentration respectively.

## Installation

The package is distributed as a standard Python project using
[PEP517](https://peps.python.org/pep-0517/).  To build and install it
locally, clone the repository and run:

```bash
pip install .[dev]
```

This command installs the package along with its optional testing
dependencies.  You need a recent version of Python (3.8 or later) and
the [JAX](https://github.com/google/jax) libraries.  Installation of
`exponax` is recommended but optional; if it is not present the solver
automatically falls back to a finite‑difference implementation.

## Usage example

```python
import jax.numpy as jnp
from acthermal_tumor.parameters import Parameters
from acthermal_tumor.solver import ThermalTumorSimulator
from acthermal_tumor.utils import generate_initial_conditions, plot_state

# Set up a 3D simulation domain
shape = (32, 32, 32)
params = Parameters(P=1.5, A=1.0, C=1.2, B=0.3, sigma_B=0.8, q=2, dt=5e-4, dx=1.0/32)
phi0, theta0, sigma0 = generate_initial_conditions(shape, radius_fraction=0.08)
sim = ThermalTumorSimulator(shape=shape, params=params)
state = sim.initialize_state(jnp.array(phi0), jnp.array(theta0), jnp.array(sigma0))

# Integrate for 200 steps
state = sim.run(state, num_steps=200)

# Visualise the central slice (automatically handled for 3D)
plot_state(state)
```

## Extending the model

The modular design of `acthermal_tumor` makes it straightforward to
swap out the numerical scheme or extend the model with additional
physics.  The `core` module defines all constitutive relations and
discrete operators; altering the form of `h`, `κ` or `F` only
requires changing those functions.  Similarly, alternative time
integrators (e.g. implicit schemes or exponential time differencing)
can be added by modifying the `ThermalTumorSimulator` class.
