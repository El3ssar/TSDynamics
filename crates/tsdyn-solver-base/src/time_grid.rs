//! Uniform time sampling from `t0` to `tf` inclusive, matching `numpy.arange`-style stepping.

#[must_use]
pub fn uniform_time_grid(t0: f64, tf: f64, dt: f64) -> Vec<f64> {
    if dt <= 0.0 {
        return vec![t0, tf];
    }
    let mut t_arr = Vec::new();
    let mut t = t0;
    while t < tf - 1e-12 {
        t_arr.push(t);
        t += dt;
    }
    if t_arr.is_empty() || (tf - *t_arr.last().unwrap()).abs() > 1e-12 {
        t_arr.push(tf);
    }
    t_arr
}

#[cfg(test)]
mod tests {
    use super::uniform_time_grid;

    #[test]
    fn grid_matches_python_like_arange() {
        let g = uniform_time_grid(0.0, 1.0, 0.3);
        assert!((g[0] - 0.0).abs() < 1e-15);
        assert!((g[g.len() - 1] - 1.0).abs() < 1e-12);
    }
}
