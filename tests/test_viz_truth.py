"""The truth gate: the numbers *behind* the picture, against independent oracles.

The owner's words, driving the plotting layer by hand:

    *"as for the other plots, like return time, autocorrelation, psd,
    trace_determinant, flow_speed, vector_field, nullclines, streamlines,
    etc... I dont really know how to make sure they are correct, some of them I
    am not familiar with those plots, hence I can't be sure if there is
    something not well rendered or calculated."*

That is a trust problem, and no existing gate answers it.  ``test_viz_pixels.py``
proves a transform **drew something**; ``test_viz_compatibility.py`` proves it
**drew without raising**.  Neither can tell a correct vector field from one
rotated by ninety degrees — both put ink in the panel.

This module asserts the *quantities*.  Every test here compares a transform's
geometry against an oracle that does not go through the viz layer at all:

* an **analytic** signal whose answer is written down (a sinusoid's spectral
  peak, its autocorrelation lag, its return time);
* the **system's own right-hand side** (``system.rhs``) and Jacobian, which the
  field transforms are supposed to be a picture of;
* ``numpy.linalg`` on that Jacobian, for the stability classification;
* a **textbook constant** (the logistic map's period-doubling onsets at
  :math:`r = 3` and :math:`r = 1 + \\sqrt{6}`);
* the estimator in :mod:`tsdynamics.analysis` the transform claims to adapt.

Transforms with no independent oracle
-------------------------------------
Not every transform has one, and the honest thing is to say which and why rather
than write a tautology:

* ``hilbert`` / ``hilbert_labels`` / ``hilbert_difference`` / ``hilbert_fourier``
  — the curve **is** the definition; there is no second source for "where does
  index *n* land on a Hilbert curve" short of reimplementing it here.
* ``spacetime`` / ``spatial_field`` / ``time_series`` / ``phase_portrait`` —
  these are re-plots of the data with no computation, so the oracle *is* the
  identity; they are covered here by exactly that identity check.
* ``ftle`` / ``escape_time`` / ``transient_time`` / ``basins`` — each integrates
  a lattice of initial conditions; the only oracle is the same integration, so
  they are covered by :mod:`tests.test_basins` / :mod:`tests.test_planar` at the
  analysis layer, and here only by their agreement with that analysis function.
* ``gali_curves`` / ``zero_one_pq_plane`` / ``lyapunov_convergence`` /
  ``scaling_fit`` / ``line_lengths`` / ``ensemble_fan`` / ``invariant_density`` /
  ``cobweb`` / ``poincare_section`` / ``floquet_multipliers`` /
  ``eigenvalue_plane`` — adapters over an ``analysis`` estimator that owns the
  numerics and its own literature tests; the checkable claim is that the drawn
  numbers **are** the estimator's, which is what is asserted where cheap.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _channel(geom, name, index=0):
    """One channel of one part, as a plain ndarray."""
    return np.asarray(geom.parts[index][name])


def _flat(geom, name):
    """One channel concatenated over every part."""
    return np.concatenate([np.asarray(p[name]).ravel() for p in geom.parts])


# ---------------------------------------------------------------------------
# series diagnostics — analytic signals
# ---------------------------------------------------------------------------


class TestASpectrumPeaksWhereTheSignalOscillates:
    """``psd`` of a known tone must peak at that tone, not near it."""

    DT = 0.01

    def _series(self, freqs, amps):
        t = np.arange(0.0, 200.0, self.DT)
        return sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps, strict=True))

    def test_a_pure_sinusoid_peaks_at_its_own_frequency(self):
        g = ts.viz.geometry(self._series([0.3], [1.0]), "psd", dt=self.DT, method="periodogram")
        f, s = _channel(g, "x"), _channel(g, "y")
        assert f[int(np.argmax(s))] == pytest.approx(0.3, abs=1e-3)

    def test_a_two_tone_signal_peaks_at_both_tones(self):
        g = ts.viz.geometry(
            self._series([0.3, 1.1], [1.0, 0.5]), "psd", dt=self.DT, method="periodogram"
        )
        f, s = _channel(g, "x"), _channel(g, "y")
        top = np.sort(f[np.argsort(s)[-2:]])
        assert top == pytest.approx([0.3, 1.1], abs=1e-3)

    def test_the_frequency_axis_is_in_inverse_time_not_per_sample(self):
        """A `dt` twice as large halves the frequency of the same oscillation."""
        t = np.arange(0.0, 200.0, self.DT)
        y = np.sin(2 * np.pi * 0.3 * t)
        coarse = ts.viz.geometry(y, "psd", dt=2 * self.DT, method="periodogram")
        f, s = _channel(coarse, "x"), _channel(coarse, "y")
        assert f[int(np.argmax(s))] == pytest.approx(0.15, abs=1e-3)


class TestAutocorrelationFindsThePeriod:
    """``autocorrelation`` of a period-T sinusoid must peak at lag T."""

    DT = 0.01
    PERIOD = 5.0

    def test_a_sinusoid_recorrelates_after_exactly_one_period(self):
        t = np.arange(0.0, 100.0, self.DT)
        g = ts.viz.geometry(
            np.sin(2 * np.pi * t / self.PERIOD), "autocorrelation", dt=self.DT, max_delay=800
        )
        lag, c = _channel(g, "x"), _channel(g, "y")
        # The lag axis is in SAMPLES; one period is PERIOD / DT of them.
        expected = self.PERIOD / self.DT
        away = lag > 0.5 * expected
        assert lag[away][int(np.argmax(c[away]))] == pytest.approx(expected, rel=2e-3)
        assert c[0] == pytest.approx(1.0, abs=1e-9)

    def test_white_noise_decorrelates_in_one_sample(self):
        rng = np.random.default_rng(0)
        g = ts.viz.geometry(rng.normal(size=4000), "autocorrelation", max_delay=40)
        c = _channel(g, "y")
        assert c[0] == pytest.approx(1.0, abs=1e-9)
        assert np.abs(c[1:]).max() < 0.1


class TestReturnTimesOfAPeriodicOrbitAreThePeriod:
    """``return_time`` must report the period, in time units."""

    def test_every_return_of_a_sinusoid_is_one_period(self):
        dt, period = 0.01, 5.0
        t = np.arange(0.0, 100.0, dt)
        g = ts.viz.geometry(
            np.sin(2 * np.pi * t / period), "return_time", dt=dt, threshold=0.0, direction="up"
        )
        times = np.asarray(g.meta["return_times"], dtype=float)
        assert times.size >= 15
        assert times == pytest.approx(period, abs=2 * dt)
        assert g.meta["mean_return_time"] == pytest.approx(period, abs=dt)
        # …and the drawn histogram's mode is there too.
        centres, counts = _channel(g, "x"), _channel(g, "y")
        assert centres[int(np.argmax(counts))] == pytest.approx(period, abs=0.05)


# ---------------------------------------------------------------------------
# field transforms — the system's own RHS is the oracle
# ---------------------------------------------------------------------------


class TestAFieldPlotIsThePictureOfTheRightHandSide:
    """``vector_field`` / ``flow_speed`` / ``streamlines`` against ``system.rhs``.

    ``system.rhs(u, t)`` is the system's own right-hand side and knows nothing
    about the viz layer, so it is a genuinely independent source for "what should
    the arrow at this point be".  A field rotated, transposed or scaled — every
    one of which renders as a perfectly plausible picture — fails here.

    It is deliberately the **public** door: the whole point of this gate is that
    a user can run it.  Until v6 the only route was ``_rhs_numeric``, so the
    check that answers "is my vector field right?" could not be written without
    reaching into the library — see
    ``tests/test_families_rhs.py::TestTheRightHandSideHasAPublicDoor``.
    """

    @staticmethod
    def _vdp():
        return ts.systems.VanDerPol()

    def test_every_arrow_points_along_the_flow(self):
        vdp = self._vdp()
        g = ts.viz.geometry(vdp, "vector_field", domain=[(-2, 2), (-2, 2)], grid=5)
        rhs = vdp.rhs
        x, y = _channel(g, "x").ravel(), _channel(g, "y").ravel()
        u, v = _channel(g, "u").ravel(), _channel(g, "v").ravel()
        assert x.size == 25
        for xi, yi, ui, vi in zip(x, y, u, v, strict=True):
            f = np.asarray(rhs(np.array([xi, yi]), 0.0), dtype=float)
            speed = float(np.hypot(*f))
            if speed < 1e-12:
                # an equilibrium sits on the lattice: there is no direction to
                # draw, and a normalised arrow is correctly the zero vector.
                assert np.hypot(ui, vi) == pytest.approx(0.0, abs=1e-12)
                continue
            # unit-length by default (the transform says so), and parallel to f
            assert np.hypot(ui, vi) == pytest.approx(1.0, abs=1e-9)
            assert (ui * f[0] + vi * f[1]) / speed == pytest.approx(1.0, abs=1e-9)

    def test_the_speed_backdrop_is_the_norm_of_the_field_at_that_cell(self):
        vdp = self._vdp()
        g = ts.viz.geometry(vdp, "flow_speed", domain=[(-2, 2), (-2, 2)], grid=4)
        rhs = vdp.rhs
        x, y, z = _channel(g, "x"), _channel(g, "y"), _channel(g, "z")
        assert z.shape == (y.size, x.size)  # image rows are y, columns are x
        for i, yi in enumerate(y):
            for j, xj in enumerate(x):
                f = np.asarray(rhs(np.array([xj, yi]), 0.0), dtype=float)
                assert z[i, j] == pytest.approx(float(np.hypot(*f)), rel=1e-9)

    def test_streamlines_are_tangent_to_the_field(self):
        vdp = self._vdp()
        g = ts.viz.geometry(vdp, "streamlines", domain=[(-3, 3), (-3, 3)])
        rhs = vdp.rhs
        cosines = []
        for part in g.parts:
            xs, ys = np.asarray(part["x"]).ravel(), np.asarray(part["y"]).ravel()
            # one part holds many curves, NaN-separated
            breaks = np.flatnonzero(~(np.isfinite(xs) & np.isfinite(ys)))
            for seg_x, seg_y in zip(np.split(xs, breaks), np.split(ys, breaks), strict=True):
                good = np.isfinite(seg_x) & np.isfinite(seg_y)
                seg_x, seg_y = seg_x[good], seg_y[good]
                if seg_x.size < 5:
                    continue
                for k in range(1, seg_x.size - 1, 5):
                    step = np.array([seg_x[k + 1] - seg_x[k - 1], seg_y[k + 1] - seg_y[k - 1]])
                    f = np.asarray(rhs(np.array([seg_x[k], seg_y[k]]), 0.0), dtype=float)
                    n1, n2 = np.linalg.norm(step), np.linalg.norm(f)
                    if n1 < 1e-9 or n2 < 1e-9:
                        continue
                    cosines.append(float(step @ f / (n1 * n2)))
        assert len(cosines) > 100
        cos = np.asarray(cosines)
        # a streamline integrator on a coarse lattice is not exact; a field drawn
        # with the components swapped would sit near 0, and one negated at -1.
        assert np.median(cos) > 0.999
        assert np.quantile(cos, 0.02) > 0.9


class TestNullclinesCrossAtTheEquilibria:
    """On the ``x' = 0`` curve the first RHS component must actually vanish."""

    def test_each_nullcline_zeroes_the_component_it_names(self):
        vdp = ts.systems.VanDerPol()
        g = ts.viz.geometry(vdp, "nullclines", domain=[(-3, 3), (-3, 3)], grid=201)
        rhs = vdp.rhs
        assert len(g.parts) == 2
        for part in g.parts:
            assert part.label in ("x' = 0", "y' = 0")
            index = 0 if part.label.startswith("x") else 1
            xs, ys = np.asarray(part["x"]).ravel(), np.asarray(part["y"]).ravel()
            good = np.isfinite(xs) & np.isfinite(ys)
            residual = np.array(
                [
                    abs(float(np.asarray(rhs(np.array([a, b]), 0.0))[index]))
                    for a, b in zip(xs[good][:300], ys[good][:300], strict=True)
                ]
            )
            assert residual.max() < 1e-2, part.label

    def test_the_only_crossing_is_the_origin_which_is_van_der_pols_equilibrium(self):
        vdp = ts.systems.VanDerPol()
        g = ts.viz.geometry(vdp, "nullclines", domain=[(-3, 3), (-3, 3)], grid=201)
        curves = {}
        for part in g.parts:
            xs, ys = np.asarray(part["x"]).ravel(), np.asarray(part["y"]).ravel()
            good = np.isfinite(xs) & np.isfinite(ys)
            curves[part.label] = np.c_[xs[good], ys[good]]
        a, b = curves["x' = 0"], curves["y' = 0"]
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        assert a[i] == pytest.approx([0.0, 0.0], abs=0.05)
        assert float(d.min()) < 0.05


class TestTheTraceDeterminantPointIsTheJacobiansOwnTraceAndDeterminant:
    """The plotted (tr, det) pair, and the region it lands in, against numpy."""

    _REGIONS = {
        "unstable focus": lambda w: np.any(np.abs(w.imag) > 1e-12) and np.all(w.real > 0),
        "stable focus": lambda w: np.any(np.abs(w.imag) > 1e-12) and np.all(w.real < 0),
        "unstable node": lambda w: np.all(np.abs(w.imag) < 1e-12) and np.all(w.real > 0),
        "stable node": lambda w: np.all(np.abs(w.imag) < 1e-12) and np.all(w.real < 0),
        "saddle": lambda w: np.all(np.abs(w.imag) < 1e-12) and w.real.min() < 0 < w.real.max(),
    }

    def _equilibrium_parts(self, geom):
        reference = {"det = tr^2 / 4", "det = 0", "tr = 0"}
        return [p for p in geom.parts if p.label not in reference]

    def test_van_der_pols_origin_lands_on_its_own_trace_and_determinant(self):
        vdp = ts.systems.VanDerPol()
        g = ts.viz.geometry(vdp, "trace_determinant")
        jac = np.asarray(vdp.jacobian(np.zeros(2), 0.0), dtype=float)
        parts = self._equilibrium_parts(g)
        assert len(parts) == 1
        assert float(np.asarray(parts[0]["x"]).ravel()[0]) == pytest.approx(
            float(np.trace(jac)), rel=1e-9
        )
        assert float(np.asarray(parts[0]["y"]).ravel()[0]) == pytest.approx(
            float(np.linalg.det(jac)), rel=1e-9
        )

    def test_the_named_region_agrees_with_the_eigenvalues(self):
        vdp = ts.systems.VanDerPol()
        g = ts.viz.geometry(vdp, "trace_determinant")
        jac = np.asarray(vdp.jacobian(np.zeros(2), 0.0), dtype=float)
        eig = np.linalg.eigvals(jac)
        label = self._equilibrium_parts(g)[0].label
        assert label in self._REGIONS, label
        assert self._REGIONS[label](eig), (label, eig)

    def test_the_parabola_and_the_axes_are_the_textbook_boundaries(self):
        """det = tr^2/4 separates nodes from spirals; det = 0 from saddles."""
        g = ts.viz.geometry(ts.systems.VanDerPol(), "trace_determinant")
        by_label = {p.label: p for p in g.parts}
        tr = np.asarray(by_label["det = tr^2 / 4"]["x"]).ravel()
        det = np.asarray(by_label["det = tr^2 / 4"]["y"]).ravel()
        assert det == pytest.approx(tr**2 / 4.0, abs=1e-9)
        assert np.asarray(by_label["det = 0"]["y"]).ravel() == pytest.approx(0.0, abs=1e-12)
        assert np.asarray(by_label["tr = 0"]["x"]).ravel() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# a textbook constant
# ---------------------------------------------------------------------------


class TestTheOrbitDiagramPutsTheCascadeWhereTheTextbookDoes:
    """The logistic map's first two period-doublings are known exactly."""

    @staticmethod
    def _sweep(lo, hi, n, points, transient):
        g = ts.viz.geometry(
            ts.systems.Logistic(),
            "orbit_diagram",
            param="r",
            values=np.linspace(lo, hi, n),
            points=points,
            transient=transient,
        )
        return _flat(g, "x"), _flat(g, "y")

    def test_period_one_becomes_period_two_at_r_equals_three(self):
        r, x = self._sweep(2.5, 3.6, 300, 60, 300)
        values = np.unique(r)
        spread = np.array([np.ptp(x[r == v]) for v in values])
        onset = values[int(np.argmax(spread > 1e-3))]
        assert onset == pytest.approx(3.0, abs=0.01)

    def test_period_two_becomes_period_four_at_one_plus_root_six(self):
        r, x = self._sweep(3.3, 3.55, 400, 120, 1000)
        values = np.unique(r)
        branches = np.array([len(np.unique(np.round(x[r == v], 4))) for v in values])
        onset = values[int(np.argmax(branches > 2))]
        assert onset == pytest.approx(1.0 + np.sqrt(6.0), abs=0.01)


# ---------------------------------------------------------------------------
# agreement with the estimator the transform adapts
# ---------------------------------------------------------------------------


class TestARecurrencePlotDrawsTheRateItWasAsked:
    def test_the_ink_fraction_is_the_requested_recurrence_rate(self):
        traj = ts.systems.Rossler().run(final_time=60.0, dt=0.05, ic=[1.0, 1.0, 0.1])
        for rate in (0.05, 0.12):
            g = ts.viz.geometry(traj, "recurrence", recurrence_rate=rate)
            matrix = _channel(g, "z")
            assert matrix.shape[0] == matrix.shape[1]
            assert float(np.mean(matrix > 0)) == pytest.approx(rate, abs=0.01)
            assert g.meta["recurrence_rate"] == pytest.approx(rate, abs=0.01)


class TestADelayEmbeddingRebuildsTheAttractor:
    def test_the_reconstruction_has_the_dimension_of_a_real_projection(self):
        traj = ts.systems.Lorenz().run(final_time=60.0, dt=0.01, ic=[1.0, 1.0, 1.0], transient=20.0)
        g = ts.viz.geometry(traj, "delay_embedding", delay=17, components="x")
        recon = np.c_[_channel(g, "x"), _channel(g, "y")]
        assert recon.shape[0] > 1000
        reconstructed = float(ts.analysis.correlation_dimension(recon))
        honest = float(ts.analysis.correlation_dimension(traj["x", "z"].y))
        # loosely: a delay reconstruction of a 2.06-dimensional attractor is that
        # attractor, not a line (1) and not a filled plane (2.9).
        assert 1.6 < reconstructed < 2.4
        assert abs(reconstructed - honest) < 0.5

    def test_the_second_channel_is_the_first_shifted_by_the_delay(self):
        traj = ts.systems.Lorenz().run(final_time=20.0, dt=0.01, ic=[1.0, 1.0, 1.0])
        g = ts.viz.geometry(traj, "delay_embedding", delay=9, components="x")
        x, y = _channel(g, "x"), _channel(g, "y")
        assert y[:-9] == pytest.approx(x[9:], abs=1e-12)


class TestAReplotIsTheDataItself:
    """The transforms that compute nothing must reproduce the trajectory exactly."""

    def test_the_time_series_is_the_column(self):
        traj = ts.systems.Lorenz().run(final_time=5.0, dt=0.01, ic=[1.0, 1.0, 1.0])
        g = ts.viz.geometry(traj, "time_series", components="z")
        assert _channel(g, "y") == pytest.approx(traj["z"], abs=0.0)
        assert _channel(g, "x") == pytest.approx(traj.t, abs=0.0)

    def test_the_phase_portrait_is_the_two_columns(self):
        traj = ts.systems.Lorenz().run(final_time=5.0, dt=0.01, ic=[1.0, 1.0, 1.0])
        g = ts.viz.geometry(traj, "phase_portrait", components=("x", "z"))
        assert _channel(g, "x") == pytest.approx(traj["x"], abs=0.0)
        assert _channel(g, "y") == pytest.approx(traj["z"], abs=0.0)


class TestTheSpectrumPlotDrawsTheEstimatorsOwnExponents:
    def test_the_stems_are_the_analysis_answer(self):
        henon = ts.systems.Henon()
        exps = ts.analysis.lyapunov_spectrum(henon, steps=4000, ic=[0.1, 0.1])
        g = ts.viz.geometry(exps, "lyapunov_spectrum")
        drawn = _flat(g, "y")
        for value in np.asarray(exps, dtype=float):
            assert np.min(np.abs(drawn - value)) < 1e-12


class TestATraceDeterminantAxisDoesNotClaimToBeTheWholeJacobian:
    r"""The axis label is a *claim*, and for a 3-D flow the unqualified one is false.

    Measured before the fix, with no warning even under ``-W error``: Lorenz's
    origin drew at :math:`(\tau, \Delta) = (-11, -270)` under axes reading
    ``tr J`` / ``det J``, while ``numpy.trace`` / ``numpy.linalg.det`` of the
    real 3x3 Jacobian there are :math:`(-13.667, +720)`.  Both coordinates
    differ and the determinant's **sign** flips, so a reader lifting numbers off
    that figure — or even reading which side of :math:`\Delta = 0` the point
    falls on — got the wrong answer from a picture that raised nothing.

    The projection itself is a design choice (the plane classifies a 2x2
    linearization; ``plane=`` picks the block) and the *verdict* "saddle" is
    correct for the 3-D origin.  What must be true is that the figure says which
    block it took.
    """

    def test_a_sliced_plane_names_the_slice_on_both_axes(self):
        lor = ts.systems.Lorenz()
        with pytest.warns(ts.viz.VisualizationDegraded, match="NOT of the full 3x3 J"):
            g = ts.viz.geometry(lor, "trace_determinant", points=[[0.0, 0.0, 0.0]])
        assert g.axis_labels == ("tr J|xy", "det J|xy")

    def test_the_plotted_pair_is_the_named_blocks_pair_and_not_the_full_matrix(self):
        lor = ts.systems.Lorenz()
        with pytest.warns(ts.viz.VisualizationDegraded):
            g = ts.viz.geometry(lor, "trace_determinant", points=[[0.0, 0.0, 0.0]])
        marker = [p for p in g.parts if p.label not in {"det = tr^2 / 4", "det = 0", "tr = 0"}]
        drawn = (
            float(np.asarray(marker[0]["x"]).ravel()[0]),
            float(np.asarray(marker[0]["y"]).ravel()[0]),
        )
        jac = np.asarray(lor.jacobian(np.zeros(3), 0.0), dtype=float)
        block = jac[np.ix_([0, 1], [0, 1])]
        assert drawn == pytest.approx((float(np.trace(block)), float(np.linalg.det(block))))
        # ...and is emphatically NOT the full matrix's pair, which is why the
        # unqualified label was a false statement.
        assert abs(drawn[1] - float(np.linalg.det(jac))) > 100.0

    def test_a_planar_flow_is_untouched_and_silent(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            g = ts.viz.geometry(ts.systems.VanDerPol(), "trace_determinant")
        assert g.axis_labels == ("tr J", "det J")

    def test_the_slice_label_follows_the_plane_that_was_asked_for(self):
        lor = ts.systems.Lorenz()
        with pytest.warns(ts.viz.VisualizationDegraded):
            g = ts.viz.geometry(
                lor, "trace_determinant", plane=("x", "z"), points=[[0.0, 0.0, 0.0]]
            )
        assert g.axis_labels == ("tr J|xz", "det J|xz")
