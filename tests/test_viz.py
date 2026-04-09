"""
Tests for tsdynamics.viz: plotters, transforms, and animators.

All tests use synthetic trajectories (conftest fixtures) — no ODE compilation.
Matplotlib is set to Agg backend in conftest.py.
"""

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_fig_ax(result):
    """Check that result is a (Figure, Axes) 2-tuple."""
    assert isinstance(result, tuple) and len(result) == 2
    fig, ax = result
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# PlotConfig
# ---------------------------------------------------------------------------


class TestPlotConfig:
    def test_defaults(self):
        from tsdynamics.viz.base import PlotConfig

        cfg = PlotConfig()
        assert cfg.figsize == (6.0, 4.0)
        assert cfg.dpi == 120
        assert cfg.tight_layout is True
        assert cfg.facecolor is None

    def test_custom_values(self):
        from tsdynamics.viz.base import PlotConfig

        cfg = PlotConfig(figsize=(10, 8), dpi=200, tight_layout=False, facecolor="black")
        assert cfg.figsize == (10, 8)
        assert cfg.dpi == 200
        assert cfg.facecolor == "black"

    def test_new_fig_ax_returns_fig_ax(self):
        from tsdynamics.viz.base import PlotConfig, new_fig_ax

        cfg = PlotConfig(figsize=(4, 3))
        result = new_fig_ax(cfg=cfg)
        assert isinstance(result[0], matplotlib.figure.Figure)

    def test_new_fig_ax_3d_projection(self):
        from tsdynamics.viz.base import new_fig_ax

        fig, ax = new_fig_ax(projection="3d")
        from mpl_toolkits.mplot3d import Axes3D

        assert isinstance(ax, Axes3D)


# ---------------------------------------------------------------------------
# Static plotters (all must return (fig, ax))
# ---------------------------------------------------------------------------


class TestStaticPlotters:
    def test_trajectory2d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import trajectory2d

        t, X = synthetic_3d_trajectory
        _is_fig_ax(trajectory2d(t, X, dims=(0, 1)))

    def test_trajectory2d_with_config(self, synthetic_3d_trajectory):
        from tsdynamics.viz.base import PlotConfig
        from tsdynamics.viz.plotters import trajectory2d

        t, X = synthetic_3d_trajectory
        _is_fig_ax(trajectory2d(t, X, dims=(1, 2), cfg=PlotConfig()))

    def test_trajectory3d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import trajectory3d

        t, X = synthetic_3d_trajectory
        _is_fig_ax(trajectory3d(t, X, dims=(0, 1, 2)))

    def test_phase_portrait(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import phase_portrait

        _, X = synthetic_3d_trajectory
        _is_fig_ax(phase_portrait(X, dims=(0, 1)))

    def test_phase_density(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import phase_density

        _, X = synthetic_3d_trajectory
        _is_fig_ax(phase_density(X, dims=(0, 1), bins=50))

    def test_poincare_scatter_2d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import poincare_scatter

        t, X = synthetic_3d_trajectory
        _is_fig_ax(poincare_scatter(t, X, section_dim=0, section_value=0.0, extract_dims=(1, 2)))

    def test_return_map(self, scalar_series):
        from tsdynamics.viz.plotters import return_map

        _, x = scalar_series
        _is_fig_ax(return_map(x, lag=1))

    def test_return_map_lag2(self, scalar_series):
        from tsdynamics.viz.plotters import return_map

        _, x = scalar_series
        _is_fig_ax(return_map(x, lag=2))

    def test_recurrence_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import recurrence_plot

        _, X = synthetic_3d_trajectory
        _is_fig_ax(recurrence_plot(X[:100], percent=10.0))

    def test_cross_recurrence_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import cross_recurrence_plot

        _, X = synthetic_3d_trajectory
        _is_fig_ax(cross_recurrence_plot(X[:80], X[:80], percent=10.0))

    def test_joint_recurrence_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz import transforms as tf
        from tsdynamics.viz.plotters import joint_recurrence_plot

        _, X = synthetic_3d_trajectory
        R1 = tf.recurrence_matrix(X[:80], percent=10.0)
        R2 = tf.recurrence_matrix(X[:80], percent=10.0)
        _is_fig_ax(joint_recurrence_plot(R1, R2))

    def test_power_spectrum_welch(self, scalar_series):
        from tsdynamics.viz.plotters import power_spectrum

        t, x = scalar_series
        dt = t[1] - t[0]
        fs = 1.0 / dt
        _is_fig_ax(power_spectrum(x, fs=fs, method="welch"))

    def test_power_spectrum_fft(self, scalar_series):
        from tsdynamics.viz.plotters import power_spectrum

        t, x = scalar_series
        dt = t[1] - t[0]
        _is_fig_ax(power_spectrum(x, fs=1.0 / dt, method="fft"))

    def test_power_spectrum_invalid_method(self, scalar_series):
        from tsdynamics.viz.plotters import power_spectrum

        _, x = scalar_series
        with pytest.raises(ValueError):
            power_spectrum(x, fs=1.0, method="invalid")

    def test_wavelet_scalogram(self, scalar_series):
        from tsdynamics.viz.plotters import wavelet_scalogram

        t, x = scalar_series
        dt = t[1] - t[0]
        _is_fig_ax(wavelet_scalogram(x, fs=1.0 / dt))

    def test_embedding_plot_2d(self, scalar_series):
        from tsdynamics.viz.plotters import embedding_plot

        _, x = scalar_series
        _is_fig_ax(embedding_plot(x, m=2, tau=5))

    def test_embedding_plot_3d(self, scalar_series):
        from tsdynamics.viz.plotters import embedding_plot

        _, x = scalar_series
        _is_fig_ax(embedding_plot(x, m=3, tau=5))

    def test_embedding_plot_m1_raises(self, scalar_series):
        from tsdynamics.viz.plotters import embedding_plot

        _, x = scalar_series
        with pytest.raises(ValueError):
            embedding_plot(x, m=1, tau=5)

    def test_pca_projection_2d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import pca_projection_plot

        _, X = synthetic_3d_trajectory
        _is_fig_ax(pca_projection_plot(X, n_components=2))

    def test_pca_projection_3d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import pca_projection_plot

        _, X = synthetic_3d_trajectory
        _is_fig_ax(pca_projection_plot(X, n_components=3))

    def test_distance_heatmap(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import distance_heatmap

        _, X = synthetic_3d_trajectory
        _is_fig_ax(distance_heatmap(X[:50]))

    def test_bifurcation_diagram(self):
        from tsdynamics.viz.plotters import bifurcation_diagram

        rng = np.random.default_rng(0)
        params = np.linspace(3.5, 4.0, 20)
        points = [rng.uniform(0.2, 0.9, size=30) for _ in params]
        _is_fig_ax(bifurcation_diagram(params, points))

    def test_lyapunov_spectrum_plot(self):
        from tsdynamics.viz.plotters import lyapunov_spectrum

        exps = np.array([0.91, 0.0, -14.57])
        _is_fig_ax(lyapunov_spectrum(exps))

    def test_kymograph_1d(self, rng):
        from tsdynamics.viz.plotters import kymograph

        T, N = 100, 20
        t = np.linspace(0, 10, T)
        Y = rng.standard_normal((T, N))
        _is_fig_ax(kymograph(t, Y, nx=N))


# ---------------------------------------------------------------------------
# Numerical transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_take_columns(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import take_columns

        _, X = synthetic_3d_trajectory
        out = take_columns(X, (0, 2))
        assert out.shape == (X.shape[0], 2)

    def test_autocorrelation_shape(self, scalar_series):
        from tsdynamics.viz.transforms import autocorrelation

        _, x = scalar_series
        acf = autocorrelation(x, max_lag=50)
        assert acf.shape == (51,)

    def test_autocorrelation_lag0_is_one(self, scalar_series):
        from tsdynamics.viz.transforms import autocorrelation

        _, x = scalar_series
        acf = autocorrelation(x, max_lag=10)
        assert acf[0] == pytest.approx(1.0)

    def test_delay_embedding_shape(self, scalar_series):
        from tsdynamics.viz.transforms import delay_embedding

        _, x = scalar_series
        E = delay_embedding(x, m=3, tau=5)
        T_eff = len(x) - (3 - 1) * 5
        assert E.shape == (T_eff, 3)

    def test_delay_embedding_m1_tau1(self, scalar_series):
        from tsdynamics.viz.transforms import delay_embedding

        _, x = scalar_series
        E = delay_embedding(x, m=1, tau=1)
        assert E.shape == (len(x), 1)

    def test_delay_embedding_raises_bad_params(self, scalar_series):
        from tsdynamics.viz.transforms import delay_embedding

        _, x = scalar_series
        with pytest.raises(ValueError):
            delay_embedding(x, m=0, tau=1)

    def test_return_map_shapes(self, scalar_series):
        from tsdynamics.viz.transforms import return_map

        _, x = scalar_series
        x0, x1 = return_map(x, lag=1)
        assert len(x0) == len(x) - 1
        assert len(x1) == len(x) - 1

    def test_return_map_lag3(self, scalar_series):
        from tsdynamics.viz.transforms import return_map

        _, x = scalar_series
        x0, x1 = return_map(x, lag=3)
        assert len(x0) == len(x) - 3

    def test_power_spectrum_fft_shape(self, scalar_series):
        from tsdynamics.viz.transforms import power_spectrum_fft

        t, x = scalar_series
        freqs, psd = power_spectrum_fft(x, fs=1.0 / (t[1] - t[0]))
        assert freqs.shape == psd.shape
        assert np.all(psd >= 0)
        assert freqs[0] == pytest.approx(0.0)

    def test_power_spectrum_welch_shape(self, scalar_series):
        from tsdynamics.viz.transforms import power_spectrum_welch

        t, x = scalar_series
        freqs, psd = power_spectrum_welch(x, fs=1.0 / (t[1] - t[0]))
        assert freqs.shape == psd.shape
        assert np.all(psd >= 0)

    def test_wavelet_scalogram_shapes(self, scalar_series):
        from tsdynamics.viz.transforms import wavelet_scalogram

        t, x = scalar_series
        t_out, scales, A = wavelet_scalogram(x, fs=1.0 / (t[1] - t[0]))
        assert t_out.shape == (len(x),)
        assert A.shape[1] == len(x)
        assert np.all(A >= 0)

    def test_recurrence_matrix_binary(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import recurrence_matrix

        _, X = synthetic_3d_trajectory
        R = recurrence_matrix(X[:100], percent=10.0)
        assert R.dtype == np.uint8
        assert set(np.unique(R)).issubset({0, 1})
        # Diagonal must be all ones (self-recurrence)
        assert np.all(np.diag(R) == 1)

    def test_recurrence_matrix_eps_param(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import recurrence_matrix

        _, X = synthetic_3d_trajectory
        R = recurrence_matrix(X[:100], eps=0.1, percent=None)
        assert R.shape == (100, 100)

    def test_cross_recurrence_matrix_shape(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import cross_recurrence_matrix

        _, X = synthetic_3d_trajectory
        R = cross_recurrence_matrix(X[:80], X[:60], percent=10.0)
        assert R.shape == (80, 60)

    def test_joint_recurrence_matrix(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import joint_recurrence_matrix, recurrence_matrix

        _, X = synthetic_3d_trajectory
        R1 = recurrence_matrix(X[:80], percent=10.0)
        R2 = recurrence_matrix(X[:80], percent=15.0)
        RJ = joint_recurrence_matrix(R1, R2)
        assert RJ.shape == (80, 80)
        # joint must be <= each individual
        assert np.all(RJ <= R1)
        assert np.all(RJ <= R2)

    def test_joint_recurrence_single_matrix(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import joint_recurrence_matrix, recurrence_matrix

        _, X = synthetic_3d_trajectory
        R = recurrence_matrix(X[:50], percent=10.0)
        RJ = joint_recurrence_matrix(R)
        np.testing.assert_array_equal(RJ, R)

    def test_joint_recurrence_raises_empty(self):
        from tsdynamics.viz.transforms import joint_recurrence_matrix

        with pytest.raises(ValueError):
            joint_recurrence_matrix()

    def test_poincare_section_returns_arrays(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import poincare_section

        t, X = synthetic_3d_trajectory
        t_hits, pts = poincare_section(t, X, section_dim=0, section_value=0.0, extract_dims=(1, 2))
        assert t_hits.ndim == 1
        if pts.size > 0:
            assert pts.shape[1] == 2

    def test_poincare_section_negative_direction(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import poincare_section

        t, X = synthetic_3d_trajectory
        t_hits, pts = poincare_section(t, X, direction="negative", extract_dims=(1, 2))
        assert t_hits.ndim == 1

    def test_peaks_returns_indices(self, scalar_series):
        from tsdynamics.viz.transforms import peaks

        _, x = scalar_series
        idx = peaks(x)
        assert idx.ndim == 1
        # All peaks must be within bounds
        assert np.all(idx >= 0) and np.all(idx < len(x))

    def test_asymptotic_samples_via_peaks(self, scalar_series):
        from tsdynamics.viz.transforms import asymptotic_samples

        _, x = scalar_series
        vals = asymptotic_samples(x, tail=0.3, via_peaks=True)
        assert vals.ndim == 1

    def test_asymptotic_samples_raw(self, scalar_series):
        from tsdynamics.viz.transforms import asymptotic_samples

        _, x = scalar_series
        vals = asymptotic_samples(x, tail=0.2, via_peaks=False, max_points=100)
        assert vals.ndim == 1
        assert len(vals) <= 100

    def test_pca_project_shapes(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import pca_project

        _, X = synthetic_3d_trajectory
        comps, var, mu = pca_project(X, n_components=2)
        assert comps.shape == (2, 3)
        assert var.shape == (2,)
        assert mu.shape == (3,)

    def test_pca_variance_nonnegative(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import pca_project

        _, X = synthetic_3d_trajectory
        _, var, _ = pca_project(X, n_components=2)
        assert np.all(var >= 0)

    def test_project_with_components(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import pca_project, project_with_components

        _, X = synthetic_3d_trajectory
        comps, _, mu = pca_project(X, n_components=2)
        Z = project_with_components(X, comps, mu)
        assert Z.shape == (X.shape[0], 2)

    def test_distance_matrix_shape(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import distance_matrix

        _, X = synthetic_3d_trajectory
        D = distance_matrix(X[:50])
        assert D.shape == (50, 50)

    def test_distance_matrix_symmetric(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import distance_matrix

        _, X = synthetic_3d_trajectory
        D = distance_matrix(X[:30])
        np.testing.assert_array_almost_equal(D, D.T)

    def test_distance_matrix_diagonal_zero(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import distance_matrix

        _, X = synthetic_3d_trajectory
        D = distance_matrix(X[:30])
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)

    def test_distance_matrix_1d_input(self, scalar_series):
        from tsdynamics.viz.transforms import distance_matrix

        _, x = scalar_series
        D = distance_matrix(x[:30])
        assert D.shape == (30, 30)

    def test_phase_space_density_shape(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import phase_space_density

        _, X = synthetic_3d_trajectory
        H, xe, ye = phase_space_density(X[:, :2], bins=20)
        assert H.shape == (20, 20)

    def test_phase_space_density_nonnegative(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import phase_space_density

        _, X = synthetic_3d_trajectory
        H, _, _ = phase_space_density(X[:, :2], bins=20)
        assert np.all(H >= 0)

    def test_phase_space_density_raises_1d(self, scalar_series):
        from tsdynamics.viz.transforms import phase_space_density

        _, x = scalar_series
        with pytest.raises(ValueError):
            phase_space_density(x[:, None])  # (T, 1) → should raise


# ---------------------------------------------------------------------------
# Animators — verify they return FuncAnimation without rendering frames
# ---------------------------------------------------------------------------


class TestAnimators:
    def test_animate_trajectory2d_returns_animation(self, synthetic_3d_trajectory):
        import matplotlib.animation as animation

        from tsdynamics.viz.animators import animate_trajectory2d

        t, X = synthetic_3d_trajectory
        anim = animate_trajectory2d(t[:100], X[:100], dims=(0, 1), interval_ms=30, trail=20)
        assert isinstance(anim, animation.FuncAnimation)

    def test_animate_trajectory3d_returns_animation(self, synthetic_3d_trajectory):
        import matplotlib.animation as animation

        from tsdynamics.viz.animators import animate_trajectory3d

        t, X = synthetic_3d_trajectory
        anim = animate_trajectory3d(t[:100], X[:100], dims=(0, 1, 2), interval_ms=30, trail=20)
        assert isinstance(anim, animation.FuncAnimation)

    def test_animate_space_time_returns_animation(self, rng):
        import matplotlib.animation as animation

        from tsdynamics.viz.animators import animate_space_time

        T, N = 80, 10
        t = np.linspace(0, 5, T)
        Y = rng.standard_normal((T, N))
        anim = animate_space_time(t, Y, interval_ms=30)
        assert isinstance(anim, animation.FuncAnimation)
