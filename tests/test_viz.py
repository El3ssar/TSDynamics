"""
Tests for tsdynamics.viz: plotters, transforms, and animators.

All tests use synthetic trajectories (conftest fixtures) — no ODE compilation.
Matplotlib is set to Agg backend in conftest.py.
"""

import matplotlib.animation
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ax=None injection contract
# ---------------------------------------------------------------------------


class TestAxInjection:
    """Verify the ax=None / ax=existing axes contract for every plotter."""

    def test_creates_axes_when_ax_is_none(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import trajectory_plot

        t, X = synthetic_3d_trajectory
        ax = trajectory_plot(t, X)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_returns_provided_ax_unchanged(self, synthetic_3d_trajectory):
        import matplotlib.pyplot as plt

        from tsdynamics.viz.plotters import trajectory_plot

        t, X = synthetic_3d_trajectory
        fig, existing_ax = plt.subplots()
        returned = trajectory_plot(t, X, ax=existing_ax)
        assert returned is existing_ax

    def test_figure_accessible_via_ax_figure(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import trajectory_plot

        t, X = synthetic_3d_trajectory
        ax = trajectory_plot(t, X)
        assert isinstance(ax.figure, matplotlib.figure.Figure)

    def test_subplot_grid_injection(self, synthetic_3d_trajectory):
        import matplotlib.pyplot as plt

        from tsdynamics.viz.plotters import trajectory_plot

        t, X = synthetic_3d_trajectory
        fig, axes = plt.subplots(1, 2)
        trajectory_plot(t, X, dims=(0, 1), ax=axes[0])
        trajectory_plot(t, X, dims=(1, 2), ax=axes[1])
        assert len(axes[0].lines) > 0
        assert len(axes[1].lines) > 0

    def test_3d_creates_axes3d_when_ax_is_none(self, synthetic_3d_trajectory):
        from mpl_toolkits.mplot3d import Axes3D

        from tsdynamics.viz.plotters import trajectory_plot_3d

        t, X = synthetic_3d_trajectory
        ax = trajectory_plot_3d(t, X)
        assert isinstance(ax, Axes3D)

    def test_kwargs_forwarded_to_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import trajectory_plot

        t, X = synthetic_3d_trajectory
        ax = trajectory_plot(t, X, color="red")
        assert ax.lines[0].get_color() == "red"


# ---------------------------------------------------------------------------
# Static plotters (all must return matplotlib.axes.Axes)
# ---------------------------------------------------------------------------


class TestStaticPlotters:
    def test_trajectory_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import trajectory_plot

        t, X = synthetic_3d_trajectory
        ax = trajectory_plot(t, X, dims=(0, 1))
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_trajectory_plot_3d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import trajectory_plot_3d

        t, X = synthetic_3d_trajectory
        ax = trajectory_plot_3d(t, X, dims=(0, 1, 2))
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_phase_portrait(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import phase_portrait

        _, X = synthetic_3d_trajectory
        ax = phase_portrait(X, dims=(0, 1))
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_phase_density(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import phase_density

        _, X = synthetic_3d_trajectory
        ax = phase_density(X, dims=(0, 1), bins=50)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_poincare_scatter_2d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import poincare_scatter

        t, X = synthetic_3d_trajectory
        ax = poincare_scatter(t, X, section_dim=0, section_value=0.0, extract_dims=(1, 2))
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_return_map_plot(self, scalar_series):
        from tsdynamics.viz.plotters import return_map_plot

        _, x = scalar_series
        ax = return_map_plot(x, lag=1)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_return_map_plot_lag2(self, scalar_series):
        from tsdynamics.viz.plotters import return_map_plot

        _, x = scalar_series
        ax = return_map_plot(x, lag=2)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_recurrence_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import recurrence_plot

        _, X = synthetic_3d_trajectory
        ax = recurrence_plot(X[:100], percent=10.0)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_cross_recurrence_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import cross_recurrence_plot

        _, X = synthetic_3d_trajectory
        ax = cross_recurrence_plot(X[:80], X[:80], percent=10.0)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_joint_recurrence_plot(self, synthetic_3d_trajectory):
        from tsdynamics.viz import recurrence_matrix
        from tsdynamics.viz.plotters import joint_recurrence_plot

        _, X = synthetic_3d_trajectory
        R1 = recurrence_matrix(X[:80], percent=10.0)
        R2 = recurrence_matrix(X[:80], percent=10.0)
        ax = joint_recurrence_plot(R1, R2)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_power_spectrum_plot_welch(self, scalar_series):
        from tsdynamics.viz.plotters import power_spectrum_plot

        t, x = scalar_series
        ax = power_spectrum_plot(x, fs=1.0 / (t[1] - t[0]), method="welch")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_power_spectrum_plot_fft(self, scalar_series):
        from tsdynamics.viz.plotters import power_spectrum_plot

        t, x = scalar_series
        ax = power_spectrum_plot(x, fs=1.0 / (t[1] - t[0]), method="fft")
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_power_spectrum_plot_invalid_method(self, scalar_series):
        from tsdynamics.viz.plotters import power_spectrum_plot

        _, x = scalar_series
        with pytest.raises(ValueError):
            power_spectrum_plot(x, fs=1.0, method="invalid")

    def test_wavelet_scalogram_plot(self, scalar_series):
        from tsdynamics.viz.plotters import wavelet_scalogram_plot

        t, x = scalar_series
        ax = wavelet_scalogram_plot(x, fs=1.0 / (t[1] - t[0]))
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_embedding_plot_2d(self, scalar_series):
        from tsdynamics.viz.plotters import embedding_plot

        _, x = scalar_series
        ax = embedding_plot(x, m=2, tau=5)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_embedding_plot_3d(self, scalar_series):
        from tsdynamics.viz.plotters import embedding_plot

        _, x = scalar_series
        ax = embedding_plot(x, m=3, tau=5)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_embedding_plot_m1_raises(self, scalar_series):
        from tsdynamics.viz.plotters import embedding_plot

        _, x = scalar_series
        with pytest.raises(ValueError):
            embedding_plot(x, m=1, tau=5)

    def test_pca_projection_2d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import pca_projection_plot

        _, X = synthetic_3d_trajectory
        ax = pca_projection_plot(X, n_components=2)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_pca_projection_3d(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import pca_projection_plot

        _, X = synthetic_3d_trajectory
        ax = pca_projection_plot(X, n_components=3)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_distance_heatmap(self, synthetic_3d_trajectory):
        from tsdynamics.viz.plotters import distance_heatmap

        _, X = synthetic_3d_trajectory
        ax = distance_heatmap(X[:50])
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_bifurcation_diagram(self):
        from tsdynamics.viz.plotters import bifurcation_diagram

        rng = np.random.default_rng(0)
        params = np.linspace(3.5, 4.0, 20)
        points = [rng.uniform(0.2, 0.9, size=30) for _ in params]
        ax = bifurcation_diagram(params, points)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_lyapunov_spectrum_plot(self):
        from tsdynamics.viz.plotters import lyapunov_spectrum_plot

        exps = np.array([0.91, 0.0, -14.57])
        ax = lyapunov_spectrum_plot(exps)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_lyapunov_spectrum_plot_dky_in_title(self):
        from tsdynamics.viz.plotters import lyapunov_spectrum_plot

        # Lorenz canonical exponents: D_KY ≈ 2 + 0.91/14.57 ≈ 2.062
        exps = np.array([0.91, 0.0, -14.57])
        ax = lyapunov_spectrum_plot(exps)
        assert "D" in ax.get_title()

    def test_lyapunov_spectrum_plot_all_negative(self):
        from tsdynamics.viz.plotters import lyapunov_spectrum_plot

        exps = np.array([-1.0, -2.0, -3.0])
        ax = lyapunov_spectrum_plot(exps)
        # D_KY = 0 when all exponents are negative
        assert "0.000" in ax.get_title()

    def test_spacetime_plot_1d(self, synthetic_ks_trajectory):
        from tsdynamics.viz.plotters import spacetime_plot

        t, X = synthetic_ks_trajectory
        ax = spacetime_plot(t, X)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_spacetime_plot_nx_mismatch_raises(self, synthetic_ks_trajectory):
        from tsdynamics.viz.plotters import spacetime_plot

        t, X = synthetic_ks_trajectory
        with pytest.raises(ValueError):
            spacetime_plot(t, X, nx=99)  # wrong nx


# ---------------------------------------------------------------------------
# Numerical transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    # --- strip_transient ---

    def test_strip_transient_by_frac(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import strip_transient

        t, X = synthetic_3d_trajectory
        t2, X2 = strip_transient(t, X, frac=0.1)
        assert len(t2) == pytest.approx(len(t) * 0.9, abs=2)
        assert X2.shape[0] == len(t2)

    def test_strip_transient_by_n(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import strip_transient

        t, X = synthetic_3d_trajectory
        t2, X2 = strip_transient(t, X, n=100)
        assert len(t2) == len(t) - 100
        assert X2.shape == (len(t) - 100, X.shape[1])

    def test_strip_transient_raises_both_given(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import strip_transient

        t, X = synthetic_3d_trajectory
        with pytest.raises(ValueError):
            strip_transient(t, X, n=10, frac=0.1)

    def test_strip_transient_default_frac(self, synthetic_3d_trajectory):
        """Calling with no n/frac args applies the default frac=0.1."""
        from tsdynamics.viz.transforms import strip_transient

        t, X = synthetic_3d_trajectory
        t2, X2 = strip_transient(t, X)
        assert len(t2) == pytest.approx(len(t) * 0.9, abs=2)

    def test_strip_transient_frac_zero_returns_all(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import strip_transient

        t, X = synthetic_3d_trajectory
        t2, X2 = strip_transient(t, X, frac=0.0)
        np.testing.assert_array_equal(t2, t)
        np.testing.assert_array_equal(X2, X)

    # --- autocorrelation ---

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

    # --- delay_embedding ---

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

    # --- return_map ---

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

    # --- power spectrum ---

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

    # --- wavelet ---

    def test_wavelet_scalogram_shapes(self, scalar_series):
        from tsdynamics.viz.transforms import wavelet_scalogram

        t, x = scalar_series
        t_out, scales, A = wavelet_scalogram(x, fs=1.0 / (t[1] - t[0]))
        assert t_out.shape == (len(x),)
        assert A.shape[1] == len(x)
        assert np.all(A >= 0)

    # --- recurrence matrices ---

    def test_recurrence_matrix_binary(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import recurrence_matrix

        _, X = synthetic_3d_trajectory
        R = recurrence_matrix(X[:100], percent=10.0)
        assert R.dtype == np.uint8
        assert set(np.unique(R)).issubset({0, 1})
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

    # --- poincare_section ---

    def test_poincare_section_returns_arrays(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import poincare_section

        t, X = synthetic_3d_trajectory
        t_hits, pts = poincare_section(t, X, section_dim=0, section_value=0.0, extract_dims=(1, 2))
        assert t_hits.ndim == 1
        if pts.size > 0:
            assert pts.shape[1] == 2

    def test_poincare_section_t_hits_nonempty(self, synthetic_3d_trajectory):
        """Spiral trajectory must have several crossings of x0=0."""
        from tsdynamics.viz.transforms import poincare_section

        t, X = synthetic_3d_trajectory
        t_hits, pts = poincare_section(t, X, section_dim=0, section_value=0.0, extract_dims=(1, 2))
        assert len(t_hits) > 0
        assert pts.shape == (len(t_hits), 2)

    def test_poincare_section_negative_direction(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import poincare_section

        t, X = synthetic_3d_trajectory
        t_hits, pts = poincare_section(t, X, direction="negative", extract_dims=(1, 2))
        assert t_hits.ndim == 1

    def test_poincare_section_invalid_direction_raises(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import poincare_section

        t, X = synthetic_3d_trajectory
        with pytest.raises(ValueError):
            poincare_section(t, X, direction="sideways")

    def test_poincare_section_no_crossings_returns_empty(self):
        """A strictly positive trajectory has no crossings of x0=0."""
        from tsdynamics.viz.transforms import poincare_section

        t = np.linspace(0, 10, 200)
        X = np.column_stack([np.ones(200) * 5, np.sin(t), t])
        t_hits, pts = poincare_section(t, X, section_dim=0, section_value=0.0)
        assert len(t_hits) == 0
        assert pts.shape == (0, 2)

    # --- peaks & asymptotic_samples ---

    def test_peaks_returns_indices(self, scalar_series):
        from tsdynamics.viz.transforms import peaks

        _, x = scalar_series
        idx = peaks(x)
        assert idx.ndim == 1
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

    # --- PCA ---

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

    def test_project_onto_pca_shape(self, synthetic_3d_trajectory):
        from tsdynamics.viz.transforms import pca_project, project_onto_pca

        _, X = synthetic_3d_trajectory
        comps, _, mu = pca_project(X, n_components=2)
        Z = project_onto_pca(X, comps, mu)
        assert Z.shape == (X.shape[0], 2)

    # --- distance matrix ---

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

    # --- phase_space_density ---

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

    def test_phase_space_density_raises_wrong_cols(self, synthetic_3d_trajectory):
        """phase_space_density requires exactly 2 columns; 3 should raise."""
        from tsdynamics.viz.transforms import phase_space_density

        _, X = synthetic_3d_trajectory
        with pytest.raises(ValueError):
            phase_space_density(X)  # shape (800, 3)

    def test_phase_space_density_raises_1d(self, scalar_series):
        from tsdynamics.viz.transforms import phase_space_density

        _, x = scalar_series
        with pytest.raises(ValueError):
            phase_space_density(x[:, None])  # shape (800, 1)


# ---------------------------------------------------------------------------
# Animators — verify they return FuncAnimation without rendering frames
# ---------------------------------------------------------------------------


class TestAnimators:
    def test_animate_trajectory_returns_animation(self, synthetic_3d_trajectory):
        from tsdynamics.viz.animators import animate_trajectory

        t, X = synthetic_3d_trajectory
        anim = animate_trajectory(t[:100], X[:100], dims=(0, 1), interval_ms=30, trail=20)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_animate_trajectory_3d_returns_animation(self, synthetic_3d_trajectory):
        from tsdynamics.viz.animators import animate_trajectory_3d

        t, X = synthetic_3d_trajectory
        anim = animate_trajectory_3d(t[:100], X[:100], dims=(0, 1, 2), interval_ms=30, trail=20)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_animate_spacetime_returns_animation(self, rng):
        from tsdynamics.viz.animators import animate_spacetime

        T, N = 80, 10
        t = np.linspace(0, 5, T)
        Y = rng.standard_normal((T, N))
        anim = animate_spacetime(t, Y, interval_ms=30)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)

    def test_animate_spacetime_high_dim(self, synthetic_ks_trajectory):
        from tsdynamics.viz.animators import animate_spacetime

        t, X = synthetic_ks_trajectory
        anim = animate_spacetime(t, X, interval_ms=50)
        assert isinstance(anim, matplotlib.animation.FuncAnimation)
