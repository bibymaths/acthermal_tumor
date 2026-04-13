"""Parameter definitions for the non‑isothermal Allen–Cahn model.

 This module defines a `Parameters` dataclass encapsulating all physical
 constants and numerical settings used in the simulator.  By grouping
 parameters into a structured object we simplify passing them between
 functions and enable type checking.
 """

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Parameters:
    """Physical and numerical parameters for the tumour growth model.

    Attributes
    ----------
    P: float
        Tumour proliferation rate.  A positive constant controlling how
        quickly tumour cells grow in response to nutrient availability.
    A: float
        Apoptosis (cell death) rate.  A positive constant that reduces
        the growth term.  When `(P * σ - A)` becomes negative the tumour
        concentration tends to decrease.
    C: float
        Nutrient consumption rate.  A positive constant multiplying
        `σ * h(φ)`.
    B: float
        Blood–tissue transfer (supply) rate.  Governs the replenishment of
        nutrient from the vasculature.
    sigma_B: float
        Nutrient concentration in the blood (vascular) reservoir.  Must lie
        between 0 and 1 in the non‑dimensional formulation.
    q: int
        Exponent used in the heat conductivity `kappa(θ) = 1 + |θ|^q`.
    dt: float
        Time‑step size for the numerical integration.
    dx: float
        Spatial grid spacing (assumed uniform in all directions).
    """

    P: float
    A: float
    C: float
    B: float
    sigma_B: float
    q: int = 2
    dt: float = 1e-3
    dx: float = 1.0 / 64

    def gradient_exponent(self) -> int:
        """Return the exponent q used in the heat conductivity.

        Included as a method to highlight that changing `q` affects the
        behaviour of the temperature equation.
        """
        return self.q
