//! Scalar I-controller for ERK step size (N2.b — PI control deferred to N2.c stiff family).

/// `err` is a normalised RMS measure (≈1 means at tolerance). Returns `(accepted, h_new)`.
#[inline]
pub fn adapt_step(
    err: f64,
    h: f64,
    order: u32,
    safety: f64,
    facmin: f64,
    facmax: f64,
) -> (bool, f64) {
    if !err.is_finite() {
        return (false, h * 0.5_f64.max(facmin).min(0.5));
    }
    let expo = 1.0 / (f64::from(order) + 1.0);
    let fac = (safety * err.powf(-expo)).clamp(facmin, facmax);
    let accepted = err <= 1.0;
    let h_new = if accepted {
        h * fac
    } else {
        h * fac.min(0.5_f64)
    };
    (accepted, h_new)
}
