//! The seeded RNG / Wiener substrate — the deterministic randomness every
//! stochastic path in the engine draws from (ROADMAP §4c).
//!
//! Determinism is a hard contract: an ensemble must produce **bit-for-bit** the
//! same result whether its trajectories run on one thread or sixteen. The
//! engine guarantees that not by serialising work but by making each
//! trajectory's randomness a pure function of `(base_seed, trajectory_index)` —
//! never of thread scheduling. [`seed_for`] is that function; the rayon fan-out
//! in [`crate::ensemble`] seeds worker `i` from it, so a stochastic kernel
//! (stream E-SDE) that owns a [`SplitMix64`] reproduces serial order exactly.
//!
//! # Why a hand-rolled generator
//!
//! The randomness contract is permanent public behaviour, so the substrate owns
//! its generator rather than delegating to an external crate whose stream could
//! change across versions. [`SplitMix64`] (Steele, Lea & Flood, *Fast splittable
//! pseudorandom number generators*, OOPSLA 2014) is small, fast, well-studied,
//! and fully specified by integer arithmetic, so its stream is identical on
//! every platform and every compiler. That is exactly what a reproducible
//! scientific result needs; the `splitmix64_seed_zero_golden_stream` test pins
//! the stream to the reference vectors so a typo in a constant fails loudly.
//!
//! # What lives here vs. in E-SDE
//!
//! E5 owns the *foundation*: the generator, the per-stream seeding function, a
//! standard-normal draw, and diagonal Wiener increments (one independent
//! `N(0, dt)` per state component — the diagonal-Itô noise of ROADMAP §11). The
//! SDE engine (stream E-SDE) builds correlated/matrix noise and the kernels that
//! consume these increments on top of this layer.

/// The SplitMix64 increment — the fractional part of the golden ratio in 2⁶⁴,
/// `⌊2⁶⁴ / φ⌋ | 1` (odd). Adding it each step makes the state cycle through all
/// 2⁶⁴ values (a full-period Weyl sequence) before the mixing stage.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First SplitMix64 mixing multiplier (the OOPSLA-2014 reference constant).
const MIX_MULT_1: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second SplitMix64 mixing multiplier (the OOPSLA-2014 reference constant).
const MIX_MULT_2: u64 = 0x94D0_49BB_1331_11EB;

/// `2⁻⁵³` — the step of the 53-bit `[0, 1)` uniform grid `f64` can represent
/// exactly. Multiplying a 53-bit integer by this gives a uniform double with no
/// rounding bias.
const F64_SCALE: f64 = 1.0 / ((1u64 << 53) as f64);

/// A SplitMix64 pseudo-random generator.
///
/// One generator belongs to one stream — typically one ensemble trajectory,
/// seeded via [`seed_for`]. It is a plain value: [`Clone`] snapshots the full
/// state (including the cached normal, see [`next_normal`](SplitMix64::next_normal))
/// so a snapshot reproduces the same continuation, and it is trivially [`Send`]
/// (each rayon worker owns its own), which is what the per-worker / no-shared-
/// state ensemble model needs.
///
/// ```
/// use tsdyn_engine::rng::SplitMix64;
/// let mut a = SplitMix64::new(42);
/// let mut b = SplitMix64::new(42);
/// // Same seed ⇒ same stream, always.
/// assert_eq!(a.next_u64(), b.next_u64());
/// ```
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    /// The Weyl-sequence state; advanced by [`GOLDEN_GAMMA`] before each mix.
    state: u64,
    /// Box–Muller produces two independent normals per pair of uniforms; the
    /// second is cached here and returned by the next
    /// [`next_normal`](SplitMix64::next_normal) call. Part of the generator
    /// state so [`Clone`] and the reproducibility contract cover it.
    cached_normal: Option<f64>,
}

impl SplitMix64 {
    /// A generator seeded with `seed`. Any `u64` is a valid seed (including 0);
    /// distinct seeds give streams with no detectable correlation.
    #[inline]
    pub fn new(seed: u64) -> Self {
        SplitMix64 {
            state: seed,
            cached_normal: None,
        }
    }

    /// The next 64 raw pseudo-random bits.
    ///
    /// Advances the Weyl state by [`GOLDEN_GAMMA`] and runs the two-multiply
    /// finalizer — the exact SplitMix64 reference algorithm.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_MULT_1);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_MULT_2);
        z ^ (z >> 31)
    }

    /// A uniform double in `[0, 1)` with 53 bits of resolution.
    ///
    /// Uses the top 53 bits of [`next_u64`](SplitMix64::next_u64) (the
    /// high bits have the best statistical quality), scaled by `2⁻⁵³`, so every
    /// representable multiple of `2⁻⁵³` is equally likely and the result is
    /// never exactly `1.0`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * F64_SCALE
    }

    /// A draw from the standard normal `N(0, 1)`.
    ///
    /// Uses the basic (trigonometric) Box–Muller transform, which converts two
    /// independent uniforms into two independent normals; the second is cached
    /// and returned by the following call, so two `next_normal` calls cost two
    /// `next_u64` calls. The first uniform is taken from `(0, 1]` (via
    /// `1 - next_f64()`) so `ln` never sees `0`.
    ///
    /// Mixing `next_normal` with `next_u64`/`next_f64` is well-defined but
    /// consumes the cached normal lazily; for a clean stream use one draw kind
    /// per generator.
    #[inline]
    pub fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        // u1 ∈ (0, 1] keeps ln(u1) finite; u2 ∈ [0, 1) spans the full angle.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (core::f64::consts::TAU * u2).sin_cos();
        self.cached_normal = Some(r * sin);
        r * cos
    }
}

/// Derive the seed for stream `index` from a run-wide `base_seed`.
///
/// This is the keystone of the parallel-equals-serial guarantee: trajectory
/// `index`'s entire random stream depends only on `(base_seed, index)`, so it is
/// independent of how many threads run or in what order rayon schedules them.
/// The mix runs `base_seed + index·γ` through the SplitMix64 finalizer, so
/// adjacent indices (the common `0, 1, 2, …` ensemble case) produce
/// well-separated, decorrelated seeds rather than nearby states.
#[inline]
pub fn seed_for(base_seed: u64, index: u64) -> u64 {
    let mut z = base_seed.wrapping_add(index.wrapping_mul(GOLDEN_GAMMA));
    z = (z ^ (z >> 30)).wrapping_mul(MIX_MULT_1);
    z = (z ^ (z >> 27)).wrapping_mul(MIX_MULT_2);
    z ^ (z >> 31)
}

/// Fill `out` with one diagonal Wiener increment per state component:
/// `dW_k ~ N(0, dt)`, i.e. `√dt · N(0, 1)`, drawn independently per component.
///
/// This is the diagonal-Itô noise substrate (ROADMAP §11): each component gets
/// its own independent Brownian increment over a step of width `dt`. `dt` must
/// be non-negative (the engine integrates forward); `dt == 0` yields all zeros.
#[inline]
pub fn fill_wiener(rng: &mut SplitMix64, dt: f64, out: &mut [f64]) {
    debug_assert!(dt >= 0.0, "Wiener step dt must be non-negative, got {dt}");
    let scale = dt.sqrt();
    for slot in out.iter_mut() {
        *slot = scale * rng.next_normal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical SplitMix64 stream for seed 0 (the reference vectors shipped
    /// with the algorithm). An independent anchor: if any magic constant above
    /// were mistyped, these would not match.
    #[test]
    fn splitmix64_seed_zero_golden_stream() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 16294208416658607535);
        assert_eq!(rng.next_u64(), 7960286522194355700);
        assert_eq!(rng.next_u64(), 487617019471545679);
    }

    #[test]
    fn same_seed_same_stream_different_seed_diverges() {
        let (mut a, mut b) = (SplitMix64::new(12345), SplitMix64::new(12345));
        let mut c = SplitMix64::new(12346);
        let xa: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let xb: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        let xc: Vec<u64> = (0..16).map(|_| c.next_u64()).collect();
        assert_eq!(xa, xb, "identical seeds must give identical streams");
        assert_ne!(xa, xc, "a one-bit seed change must change the stream");
    }

    #[test]
    fn next_f64_is_in_unit_interval() {
        let mut rng = SplitMix64::new(99);
        for _ in 0..100_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "uniform out of [0, 1): {x}");
        }
    }

    #[test]
    fn next_f64_mean_is_about_half() {
        let mut rng = SplitMix64::new(7);
        let n = 200_000;
        let mean = (0..n).map(|_| rng.next_f64()).sum::<f64>() / n as f64;
        // Std error of the mean of U(0,1) is 1/sqrt(12 n) ≈ 6.5e-4 here; 0.01
        // is a very loose, non-flaky bound.
        assert!((mean - 0.5).abs() < 0.01, "uniform mean off: {mean}");
    }

    #[test]
    fn next_normal_matches_standard_moments() {
        let mut rng = SplitMix64::new(2024);
        let n = 400_000;
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for _ in 0..n {
            let z = rng.next_normal();
            sum += z;
            sumsq += z * z;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "normal mean off: {mean}");
        assert!((var - 1.0).abs() < 0.02, "normal variance off: {var}");
    }

    #[test]
    fn cached_normal_survives_clone() {
        // After an odd number of draws a partner normal is cached; a clone must
        // continue the *identical* stream, cache included.
        let mut a = SplitMix64::new(555);
        let _ = a.next_normal(); // leaves one cached
        let mut b = a.clone();
        assert_eq!(a.next_normal(), b.next_normal());
        assert_eq!(a.next_normal(), b.next_normal());
    }

    #[test]
    fn seed_for_is_pure_and_well_separated() {
        // Pure function of its inputs …
        assert_eq!(seed_for(42, 7), seed_for(42, 7));
        // … and adjacent indices land far apart (no obvious low-bit structure).
        let seeds: Vec<u64> = (0..1000).map(|i| seed_for(0xDEAD_BEEF, i)).collect();
        let mut sorted = seeds.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), seeds.len(), "seed_for collided on 0..1000");
    }

    #[test]
    fn fill_wiener_scales_as_sqrt_dt() {
        let mut rng = SplitMix64::new(1);
        let dt = 0.01;
        let n = 100_000;
        let mut buf = [0.0; 1];
        let mut sumsq = 0.0;
        for _ in 0..n {
            fill_wiener(&mut rng, dt, &mut buf);
            sumsq += buf[0] * buf[0];
        }
        // E[dW²] = dt; the empirical second moment should land near it.
        let est = sumsq / n as f64;
        assert!(
            (est - dt).abs() < 0.1 * dt,
            "Wiener variance off: {est} vs {dt}"
        );
    }

    #[test]
    fn fill_wiener_zero_dt_is_zero() {
        let mut rng = SplitMix64::new(1);
        let mut buf = [9.0; 4];
        fill_wiener(&mut rng, 0.0, &mut buf);
        assert_eq!(buf, [0.0; 4]);
    }
}
