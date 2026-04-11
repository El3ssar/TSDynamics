"""
tsdynamics.viz — visualization submodule.

All public functions are importable directly from this namespace::

    from tsdynamics.viz import phase_portrait, poincare_section, strip_transient

Module layout
-------------
transforms.py
    Pure numpy/scipy functions: array → array.  No matplotlib.
plotters.py
    Static plot functions.  Each accepts ``ax=None`` and returns ``ax``.
animators.py
    ``FuncAnimation`` wrappers.  Each accepts ``ax=None`` and returns an animation.
_utils.py
    Private helpers (_resolve_ax, _label_dims, Protocol stubs).  Not public API.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Animators
# ---------------------------------------------------------------------------
from .animators import (
    animate_spacetime,
    animate_trajectory,
    animate_trajectory_3d,
)

# ---------------------------------------------------------------------------
# Static plotters
# ---------------------------------------------------------------------------
from .plotters import (
    bifurcation_diagram,
    cross_recurrence_plot,
    distance_heatmap,
    embedding_plot,
    joint_recurrence_plot,
    lyapunov_spectrum_plot,
    pca_projection_plot,
    phase_density,
    phase_portrait,
    poincare_scatter,
    power_spectrum_plot,
    recurrence_plot,
    return_map_plot,
    spacetime_plot,
    trajectory_plot,
    trajectory_plot_3d,
    wavelet_scalogram_plot,
)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
from .transforms import (
    asymptotic_samples,
    autocorrelation,
    cross_recurrence_matrix,
    delay_embedding,
    distance_matrix,
    joint_recurrence_matrix,
    pca_project,
    peaks,
    phase_space_density,
    poincare_section,
    power_spectrum_fft,
    power_spectrum_welch,
    project_onto_pca,
    recurrence_matrix,
    return_map,
    strip_transient,
    wavelet_scalogram,
)

__all__ = [
    # transforms
    "asymptotic_samples",
    "autocorrelation",
    "cross_recurrence_matrix",
    "delay_embedding",
    "distance_matrix",
    "joint_recurrence_matrix",
    "peaks",
    "pca_project",
    "phase_space_density",
    "poincare_section",
    "power_spectrum_fft",
    "power_spectrum_welch",
    "project_onto_pca",
    "recurrence_matrix",
    "return_map",
    "strip_transient",
    "wavelet_scalogram",
    # plotters
    "bifurcation_diagram",
    "cross_recurrence_plot",
    "distance_heatmap",
    "embedding_plot",
    "joint_recurrence_plot",
    "lyapunov_spectrum_plot",
    "pca_projection_plot",
    "phase_density",
    "phase_portrait",
    "poincare_scatter",
    "power_spectrum_plot",
    "recurrence_plot",
    "return_map_plot",
    "spacetime_plot",
    "trajectory_plot",
    "trajectory_plot_3d",
    "wavelet_scalogram_plot",
    # animators
    "animate_spacetime",
    "animate_trajectory",
    "animate_trajectory_3d",
]
