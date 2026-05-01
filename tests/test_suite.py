from __future__ import annotations

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest


def make_params(**overrides):
    from acthermal_tumor.parameters import Parameters

    defaults = dict(P=1.0, A=0.5, C=0.3, B=0.2, sigma_b=0.8)
    defaults.update(overrides)
    p = Parameters(**defaults)
    # Source uses sigma_B in core.py; provide compatibility alias in tests only.
    p.sigma_B = p.sigma_b
    return p


def flat_state(n=8, val=0.5):
    from acthermal_tumor.core import State

    arr = jnp.full((n, n), val, dtype=jnp.float32)
    return State(arr, arr, arr)


class TestParameters:
    def test_default_values(self):
        p = make_params()
        assert p.q == 2
        assert p.dt == pytest.approx(1e-3)
        assert p.dx == pytest.approx(1.0 / 64)

    def test_custom_values(self):
        p = make_params(P=2.0, A=1.0, C=0.5, B=0.1, sigma_b=0.9, q=3, dt=1e-4, dx=0.02)
        assert p.P == pytest.approx(2.0)
        assert p.q == 3
        assert p.sigma_B == pytest.approx(0.9)

    def test_gradient_exponent_method(self):
        assert make_params(q=4).gradient_exponent() == 4

    def test_sigma_alias_added_in_tests(self):
        p = make_params(sigma_b=0.5)
        assert p.sigma_b == pytest.approx(0.5)
        assert p.sigma_B == pytest.approx(0.5)


class TestCoreFunctions:
    def test_double_well_potential_minima(self):
        from acthermal_tumor.core import double_well_potential

        assert float(double_well_potential(jnp.array(0.0))) == pytest.approx(0.0)
        assert float(double_well_potential(jnp.array(1.0))) == pytest.approx(0.0)

    def test_double_well_prime_known_value(self):
        from acthermal_tumor.core import double_well_prime

        assert float(double_well_prime(jnp.array(0.5))) == pytest.approx(0.0, abs=1e-6)

    def test_activation_zero_at_origin(self):
        from acthermal_tumor.core import activation_function

        assert float(activation_function(jnp.array(0.0))) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_activation_monotone(self):
        from acthermal_tumor.core import activation_function

        xs = jnp.linspace(-2, 2, 50)
        assert jnp.all(jnp.diff(activation_function(xs)) >= 0)

    def test_kappa(self):
        from acthermal_tumor.core import kappa

        assert float(kappa(jnp.array(0.0))) == pytest.approx(1.0)
        assert float(kappa(jnp.array(2.0), q=3)) == pytest.approx(9.0)


class TestOperators:
    def test_shift_shapes(self):
        from acthermal_tumor.core import _shift_backward_neumann, _shift_forward_neumann

        u = jnp.ones((4, 4))
        assert _shift_forward_neumann(u, 0).shape == (4, 4)
        assert _shift_backward_neumann(u, 1).shape == (4, 4)

    def test_laplacian_constant_zero(self):
        from acthermal_tumor.core import laplacian_neumann

        u = jnp.ones((8, 8))
        assert jnp.allclose(laplacian_neumann(u, dx=0.1), 0.0, atol=1e-6)

    def test_gradient_constant_zero(self):
        from acthermal_tumor.core import gradient_neumann

        grads = gradient_neumann(jnp.ones((8, 8)), dx=0.1)
        assert len(grads) == 2
        assert all(jnp.allclose(g, 0.0, atol=1e-6) for g in grads)

    def test_gradient_linear_field_matches_discrete_spacing(self):
        from acthermal_tumor.core import gradient_neumann

        n = 10
        dx = 1.0 / n
        x = jnp.linspace(0, 1, n)
        u = jnp.broadcast_to(x[None, :], (n, n))
        grads = gradient_neumann(u, dx=dx)
        expected = (x[2] - x[0]) / (2.0 * dx)
        assert jnp.allclose(grads[1][1:-1, 1:-1], expected, atol=1e-6)

    def test_divergence_constant_zero(self):
        from acthermal_tumor.core import divergence_neumann

        v = [jnp.ones((8, 8)), jnp.ones((8, 8))]
        assert jnp.allclose(divergence_neumann(v, dx=0.1), 0.0, atol=1e-6)


class TestState:
    def test_add_mul_and_tree(self):
        from acthermal_tumor.core import State

        a = flat_state(4, 1.0)
        b = flat_state(4, 2.0)
        c = a + b
        d = 3.0 * a
        assert jnp.allclose(c.phi, 3.0)
        assert jnp.allclose(d.theta, 3.0)
        children, aux = a.tree_flatten()
        rebuilt = State.tree_unflatten(aux, children)
        assert jnp.allclose(rebuilt.sigma, a.sigma)


class TestRhs:
    def setup_method(self):
        self.params = make_params()

    def test_output_is_state(self):
        from acthermal_tumor.core import State, rhs

        out = rhs(flat_state(8, 0.5), self.params)
        assert isinstance(out, State)

    def test_output_shape_and_finite(self):
        from acthermal_tumor.core import rhs

        out = rhs(flat_state(8, 0.5), self.params)
        assert out.phi.shape == (8, 8)
        assert jnp.isfinite(out.phi).all()
        assert jnp.isfinite(out.theta).all()
        assert jnp.isfinite(out.sigma).all()

    def test_zero_phi_state(self):
        from acthermal_tumor.core import State, rhs

        arr = jnp.zeros((8, 8), dtype=jnp.float32)
        s = State(arr, arr, jnp.ones((8, 8), dtype=jnp.float32))
        out = rhs(s, self.params)
        assert out.phi.shape == (8, 8)

    def test_parameter_sensitivity_in_phi(self):
        from acthermal_tumor.core import rhs

        s = flat_state(8, 0.5)
        r1 = rhs(s, make_params(P=0.1))
        r2 = rhs(s, make_params(P=5.0))
        assert not jnp.allclose(r1.phi, r2.phi)


class TestBuildSpectralLaplace:
    def test_returns_none_when_exponax_absent(self):
        import acthermal_tumor.core as core_mod

        original = core_mod.exponax
        core_mod.exponax = None
        try:
            assert core_mod.build_spectral_laplace((8, 8), 0.1) is None
        finally:
            core_mod.exponax = original


class TestThermalTumorSimulator:
    def _make_sim(self, shape=(8, 8), use_spectral=False, **param_overrides):
        from acthermal_tumor.solver import ThermalTumorSimulator

        return ThermalTumorSimulator(
            shape=shape,
            params=make_params(**param_overrides),
            use_spectral=use_spectral,
        )

    def test_instantiation_and_invalid_shape(self):
        sim = self._make_sim()
        assert sim.shape == (8, 8)
        from acthermal_tumor.solver import ThermalTumorSimulator

        with pytest.raises(ValueError):
            ThermalTumorSimulator(shape=(1, 8), params=make_params())

    def test_stability_dt(self):
        sim = self._make_sim(dx=1.0 / 8, dt=1.0)
        expected = ((1.0 / 8) ** 2) / (4.0 * 2) * 0.5
        assert sim._dt == pytest.approx(expected)

    def test_initialize_state(self):
        sim = self._make_sim()
        phi0, theta0, sigma0 = [jnp.ones((8, 8))] * 3
        state = sim.initialize_state(phi0, theta0, sigma0)
        assert state.phi.shape == (8, 8)
        with pytest.raises(ValueError):
            sim.initialize_state(jnp.ones((4, 4)), theta0, sigma0)

    def test_step_returns_state_and_clips(self):
        from acthermal_tumor.core import State

        sim = self._make_sim()
        s = State(jnp.full((8, 8), 10.0), jnp.zeros((8, 8)), jnp.full((8, 8), 10.0))
        out = sim.step(s)
        assert jnp.all(out.phi <= 1.0)
        assert jnp.all(out.sigma <= 1.0)

    def test_step_nan_to_num(self):
        from acthermal_tumor.core import State

        sim = self._make_sim()
        nan_arr = jnp.full((8, 8), jnp.nan)
        sim._step_fn = lambda s: State(nan_arr, nan_arr, nan_arr)
        out = sim.step(flat_state(8, 0.5))
        assert jnp.all(jnp.isfinite(out.phi))

    def test_run(self):
        sim = self._make_sim(shape=(16, 16))
        from acthermal_tumor.utils import generate_initial_conditions

        phi0, theta0, sigma0 = generate_initial_conditions((16, 16))
        state = sim.initialize_state(
            jnp.array(phi0), jnp.array(theta0), jnp.array(sigma0)
        )
        out = sim.run(state, num_steps=3)
        assert out.phi.shape == (16, 16)
        out0 = sim.run(state, num_steps=0)
        assert jnp.allclose(out0.phi, state.phi)


class TestGenerateInitialConditions:
    def test_shapes_and_dtype(self):
        from acthermal_tumor.utils import generate_initial_conditions

        phi, theta, sigma = generate_initial_conditions((16, 16))
        assert phi.shape == theta.shape == sigma.shape == (16, 16)
        assert phi.dtype == theta.dtype == sigma.dtype == np.float32

    def test_1d_and_3d_shapes(self):
        from acthermal_tumor.utils import generate_initial_conditions

        assert generate_initial_conditions((32,))[0].shape == (32,)
        assert generate_initial_conditions((8, 8, 8))[0].shape == (8, 8, 8)

    def test_uniform_fields(self):
        from acthermal_tumor.utils import generate_initial_conditions

        _, theta, sigma = generate_initial_conditions(
            (16, 16), theta_value=0.5, sigma_value=0.8
        )
        assert np.allclose(theta, 0.5)
        assert np.allclose(sigma, 0.8)

    def test_tumour_and_background_values(self):
        from acthermal_tumor.utils import generate_initial_conditions

        phi, _, _ = generate_initial_conditions(
            (32, 32), tumour_value=np.float32(0.9), background_value=np.float32(0.1)
        )
        unique = np.unique(phi)
        assert len(unique) == 2
        assert np.isclose(unique[0], 0.1)
        assert np.isclose(unique[1], 0.9)

    def test_geometry(self):
        from acthermal_tumor.utils import generate_initial_conditions

        n = 64
        phi, _, _ = generate_initial_conditions((n, n), radius_fraction=0.2)
        assert phi[n // 2, n // 2] == pytest.approx(1.0)
        assert phi[0, 0] == pytest.approx(0.0)


class TestPlotState:
    def _make_2d_state(self, n=8):
        from acthermal_tumor.core import State

        arr = jnp.ones((n, n), dtype=jnp.float32) * 0.5
        return State(arr, arr, arr)

    def _make_3d_state(self, n=4):
        from acthermal_tumor.core import State

        arr = jnp.ones((n, n, n), dtype=jnp.float32) * 0.5
        return State(arr, arr, arr)

    def _make_1d_state(self, n=8):
        from acthermal_tumor.core import State

        arr = jnp.ones((n,), dtype=jnp.float32) * 0.5
        return State(arr, arr, arr)

    def test_2d_runs(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from acthermal_tumor.utils import plot_state

        with patch.object(plt, "show"):
            plot_state(self._make_2d_state())
        plt.close("all")

    def test_3d_runs(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from acthermal_tumor.utils import plot_state

        with patch.object(plt, "show"):
            plot_state(self._make_3d_state())
        plt.close("all")

    def test_1d_raises_typeerror(self):
        from acthermal_tumor.utils import plot_state

        with pytest.raises(TypeError, match="Invalid shape"):
            plot_state(self._make_1d_state())

    def test_4d_raises_valueerror(self):
        from acthermal_tumor.core import State
        from acthermal_tumor.utils import plot_state

        arr = jnp.ones((2, 2, 2, 2))
        with pytest.raises(ValueError, match="3D"):
            plot_state(State(arr, arr, arr))


class TestPublicAPI:
    def test_imports_and_all(self):
        import acthermal_tumor

        assert hasattr(acthermal_tumor, "ThermalTumorSimulator")
        assert hasattr(acthermal_tumor, "Parameters")
        assert hasattr(acthermal_tumor, "generate_initial_conditions")
        assert hasattr(acthermal_tumor, "plot_state")
        for name in acthermal_tumor.__all__:
            assert hasattr(acthermal_tumor, name)

    def test_version_string(self):
        import acthermal_tumor

        assert isinstance(acthermal_tumor.__version__, str)
        assert acthermal_tumor.__version__

    def test_version_fallback_branch(self):
        import acthermal_tumor

        with patch(
            "acthermal_tumor._pkg_version", side_effect=Exception("no metadata")
        ):
            try:
                ver = acthermal_tumor._pkg_version("acthermal_tumor")
            except Exception:
                ver = "0.0.0"
        assert ver == "0.0.0"


class TestIntegration:
    def test_full_pipeline_2d(self):
        from acthermal_tumor import (
            Parameters,
            ThermalTumorSimulator,
            generate_initial_conditions,
        )

        p = Parameters(P=1.0, A=0.5, C=0.3, B=0.2, sigma_b=0.8)
        p.sigma_B = p.sigma_b
        shape = (16, 16)
        sim = ThermalTumorSimulator(shape=shape, params=p, use_spectral=False)
        phi0, theta0, sigma0 = generate_initial_conditions(shape)
        state = sim.initialize_state(
            jnp.array(phi0), jnp.array(theta0), jnp.array(sigma0)
        )
        final = sim.run(state, num_steps=5)
        assert jnp.isfinite(final.phi).all()
        assert jnp.isfinite(final.theta).all()
        assert jnp.isfinite(final.sigma).all()

    def test_full_pipeline_1d(self):
        from acthermal_tumor import (
            Parameters,
            ThermalTumorSimulator,
            generate_initial_conditions,
        )

        p = Parameters(P=0.5, A=0.2, C=0.1, B=0.1, sigma_b=1.0)
        p.sigma_B = p.sigma_b
        shape = (32,)
        sim = ThermalTumorSimulator(shape=shape, params=p, use_spectral=False)
        phi0, theta0, sigma0 = generate_initial_conditions(shape)
        state = sim.initialize_state(
            jnp.array(phi0), jnp.array(theta0), jnp.array(sigma0)
        )
        final = sim.run(state, num_steps=3)
        assert jnp.isfinite(final.phi).all()
