"""Explicit return contracts for analyses outside the AnalysisResult hierarchy.

These are supported API shapes, not expected failures.  Pin the complete
annotation so migrating an analysis to a result wrapper requires removing its
exception rather than continuing to skip the result contract.
"""

import numpy as np

from tsdynamics.analysis.dimensions._common import DimensionResult
from tsdynamics.analysis.dimensions.correlation import CorrelationSum
from tsdynamics.analysis.planar import FlowField, Nullcline, ScalarField, TraceDeterminant
from tsdynamics.analysis.sampling.sagitta import SagittaDt

NON_RESULT_RETURNS: dict[str, object] = {
    "autocorrelation": np.ndarray,
    "correlation_sum": CorrelationSum,
    "dimension_spectrum": dict[float, DimensionResult],
    "escape_time_field": ScalarField,
    "estimate_dt_from_sagitta": SagittaDt,
    "flow_field": FlowField,
    "ftle_field": ScalarField,
    "invariant_density": tuple[np.ndarray, np.ndarray, np.ndarray],
    "nullclines": list[Nullcline],
    "sagitta_profile": np.ndarray,
    "set_distance": float,
    "streamlines": list[np.ndarray],
    "trace_determinant": TraceDeterminant,
    "transient_time_field": ScalarField,
}
