//! A process-wide, bounded, thread-safe cache of **compiled** evaluators.
//!
//! # Why
//!
//! [`JitEvaluator::new`] runs Cranelift over the whole tape on every call. That
//! is a *per-FFI-call* cost, and the bridge builds a fresh evaluator for every
//! `integrate` / `iterate` / `ensemble` entry, so a `backend="jit"` run pays the
//! full compile before it takes its first step. Measured on this tree:
//! ~0.13 ms for Lorenz (3 states, ~40 instructions) but ~296 ms for Gray–Scott
//! (a 2 × 32 × 32 method-of-lines field, ~28 k instructions). The consequences
//! were that `jit` was *slower* than the interpreter below ~1000 Lorenz steps,
//! and that a Gray–Scott parameter sweep re-compiled a byte-identical tape once
//! per value.
//!
//! Python already solves the analogous problem one layer up —
//! `tsdynamics._engine.compile` memoises *lowered* tapes in a bounded LRU with
//! `clear_tape_cache()` / `tape_cache_stats()` and a `TSDYNAMICS_NO_TAPE_CACHE`
//! bypass. This is the same design for the *compiled* artifact, so a repeat call
//! on an unchanged tape compiles once.
//!
//! # Correctness: the key is the tape itself
//!
//! A stale hit would silently return wrong numbers, so the cache does not hash
//! a summary of the tape and trust it: the hash only chooses a candidate, and a
//! candidate is accepted only on **full [`Tape`] equality** (`PartialEq` over
//! `ops`, `a`, `b`, `imm`, `outputs`, `jac_outputs`, `n_state`, `n_param` — the
//! complete input to [`crate::codegen::compile`], hence to the generated code).
//! A hash collision therefore costs a compile, never a wrong answer.
//!
//! # Bounds and concurrency
//!
//! The store is a `Vec` of at most [`CACHE_MAXSIZE`] entries under one
//! `Mutex`, evicting least-recently-used. Compilation happens **outside** the
//! lock (two threads racing on the same new tape both compile; the loser drops
//! its copy and takes the winner's, so at most one evaluator per tape is
//! retained). Setting [`CACHE_DISABLE_ENV`] truthy turns the cache into a
//! straight passthrough, which is what lets a test prove WITH-cache ==
//! WITHOUT-cache.
//!
//! [`clear_cache`] is a **barrier**: an evaluator whose compile started before
//! a clear is handed back to its caller but not retained (the `generation`
//! counter), so a clear cannot be undone by an in-flight compile landing after
//! it.
//!
//! # Forking
//!
//! This cache shares the residual `tsdyn_engine::pool` documents under "The one
//! case that stays broken (and always will)": forking while another thread
//! holds this `Mutex` copies it locked, and the child would block on it. The
//! window here is strictly narrower than the pool's — the lock covers only a
//! scan of ≤ [`CACHE_MAXSIZE`] entries, never the Cranelift compile — and,
//! unlike the pool, there is nothing to repair in the child: the inherited
//! compiled pages are valid there. `fork()` in a multithreaded process is
//! defined only if the child goes straight to `exec()`, and the case anyone
//! actually hits (`multiprocessing` forking from an idle parent) is safe; see
//! `tests/test_engine_process_safety.py`.

use std::sync::{Arc, Mutex, MutexGuard};

use tsdyn_ir::Tape;

use crate::error::JitError;
use crate::evaluator::JitEvaluator;

/// Maximum number of distinct compiled evaluators retained (LRU eviction).
///
/// Deliberately smaller than the Python tape cache's 256: an entry here holds
/// *executable* pages plus the Cranelift module that owns them, which is far
/// heavier per entry than a lowered tape. A session touches one entry per
/// distinct `(system, with_jacobian)` pair — a parameter sweep, a continuation
/// or an ensemble all key on a single entry — so 64 covers realistic use with
/// room to spare.
pub const CACHE_MAXSIZE: usize = 64;

/// Env var: set truthy (`1`/`true`/`yes`/`on`) to disable the cache
/// process-wide, so every call compiles afresh.
pub const CACHE_DISABLE_ENV: &str = "TSDYNAMICS_NO_JIT_CACHE";

/// Snapshot of the cache counters — the mirror of Python's
/// `tape_cache_stats()`, and what a test asserts on to prove a repeat call was
/// actually served from the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCacheStats {
    /// Calls served by an existing compiled evaluator.
    pub hits: u64,
    /// Calls that had to compile.
    pub misses: u64,
    /// Entries currently retained.
    pub size: usize,
    /// The retention bound ([`CACHE_MAXSIZE`]).
    pub maxsize: usize,
}

struct Entry {
    hash: u64,
    tape: Tape,
    ev: Arc<JitEvaluator>,
    /// Logical clock value of this entry's last use (LRU ordering).
    stamp: u64,
}

struct Cache {
    entries: Vec<Entry>,
    hits: u64,
    misses: u64,
    clock: u64,
    /// Bumped by [`clear_cache`]. A compile that started before a clear must not
    /// repopulate the cache after it (see `cached_evaluator`).
    generation: u64,
}

impl Cache {
    const fn new() -> Self {
        Cache {
            entries: Vec::new(),
            hits: 0,
            misses: 0,
            clock: 0,
            generation: 0,
        }
    }

    /// Find `tape`'s compiled evaluator, refreshing its LRU stamp.
    ///
    /// The `hash` pre-filter is only a fast reject; acceptance is full tape
    /// equality (see the module docs).
    fn lookup(&mut self, hash: u64, tape: &Tape) -> Option<Arc<JitEvaluator>> {
        let pos = self
            .entries
            .iter()
            .position(|e| e.hash == hash && &e.tape == tape)?;
        self.clock += 1;
        self.entries[pos].stamp = self.clock;
        Some(Arc::clone(&self.entries[pos].ev))
    }

    fn insert(&mut self, hash: u64, tape: Tape, ev: Arc<JitEvaluator>) {
        self.clock += 1;
        self.entries.push(Entry {
            hash,
            tape,
            ev,
            stamp: self.clock,
        });
        while self.entries.len() > CACHE_MAXSIZE {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.stamp)
                .map(|(i, _)| i)
                .expect("non-empty: len > CACHE_MAXSIZE >= 1");
            self.entries.swap_remove(oldest);
        }
    }
}

static CACHE: Mutex<Cache> = Mutex::new(Cache::new());

/// Lock the store, recovering from a poisoned mutex.
///
/// A panic while the lock is held would poison it and turn every later engine
/// call into a panic. The guarded state is a plain LRU of validated tapes and
/// finished evaluators — no invariant a partial update could break — so taking
/// the inner value is the right recovery, not a `unwrap()` that spreads the
/// failure.
fn lock() -> MutexGuard<'static, Cache> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the cache is active (off when [`CACHE_DISABLE_ENV`] is truthy).
///
/// Read per call, like Python's `_cache_enabled()`, so a test can flip the
/// bypass at runtime. `var_os` + a byte compare avoids allocating.
fn cache_enabled() -> bool {
    match std::env::var_os(CACHE_DISABLE_ENV) {
        None => true,
        Some(v) => {
            let s = v.to_string_lossy().trim().to_ascii_lowercase();
            !matches!(s.as_str(), "1" | "true" | "yes" | "on")
        }
    }
}

/// FNV-1a over every field that reaches codegen.
///
/// Only a bucketing hash — [`Cache::lookup`] confirms with `Tape::eq`, so a
/// collision costs a redundant compile, never a stale hit. Immediates are hashed
/// by [`f64::to_bits`], which is *finer* than `Tape`'s float equality: two tapes
/// differing only by `0.0` vs `-0.0` hash apart and so simply both compile, and
/// a tape carrying a `NaN` immediate never compares equal to itself and so never
/// caches. Both are lost reuse, not wrong code — the direction an error here has
/// to fall.
fn tape_hash(tape: &Tape) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(tape.n_state() as u64);
    mix(tape.n_param() as u64);
    mix(tape.ops().len() as u64);
    for op in tape.ops() {
        mix(op.to_i32() as u32 as u64);
    }
    for (&a, &b) in tape.a().iter().zip(tape.b()) {
        mix(((a as u32 as u64) << 32) | (b as u32 as u64));
    }
    for &v in tape.imm() {
        mix(v.to_bits());
    }
    mix(tape.outputs().len() as u64);
    for &o in tape.outputs() {
        mix(o as u32 as u64);
    }
    mix(tape.jac_outputs().len() as u64);
    for &o in tape.jac_outputs() {
        mix(o as u32 as u64);
    }
    h
}

/// Return `tape`'s compiled evaluator, compiling it only on a cache miss.
///
/// The shared-ownership return ([`Arc`]) is what lets one compiled artifact back
/// many concurrent runs; [`crate::SharedJitEvaluator`] wraps it back into an
/// [`Evaluator`](tsdyn_ir::Evaluator) for the engine's `&dyn Evaluator` seam.
///
/// Numerically this is exactly [`JitEvaluator::new`] — the same tape compiles to
/// the same code — so the `interp == jit` bit-for-bit contract is untouched.
pub fn cached_evaluator(tape: &Tape) -> Result<Arc<JitEvaluator>, JitError> {
    if !cache_enabled() {
        return Ok(Arc::new(JitEvaluator::new(tape)?));
    }
    let hash = tape_hash(tape);
    let generation = {
        let mut cache = lock();
        if let Some(ev) = cache.lookup(hash, tape) {
            cache.hits += 1;
            return Ok(ev);
        }
        cache.misses += 1;
        cache.generation
    };
    // Compile outside the lock: Cranelift takes ~0.1 ms–0.3 s, far too long to
    // hold a process-wide mutex that every engine call passes through.
    let ev = Arc::new(JitEvaluator::new(tape)?);
    #[cfg(test)]
    tests::run_compile_window_hook();
    let mut cache = lock();
    // Another thread may have compiled the same tape while we did; prefer its
    // entry so the cache keeps at most one evaluator per tape. Not a "hit" —
    // this call already compiled, and the miss is already counted.
    if let Some(existing) = cache.lookup(hash, tape) {
        return Ok(existing);
    }
    // A `clear_cache()` landed while we compiled. Clearing is a barrier — it
    // promises the cache is empty and that the pages it held are released — so
    // hand back the evaluator we just built without retaining it, rather than
    // resurrecting an entry the caller asked to be gone.
    if cache.generation != generation {
        return Ok(ev);
    }
    cache.insert(hash, tape.clone(), Arc::clone(&ev));
    Ok(ev)
}

/// Drop every cached evaluator and reset the hit/miss counters.
///
/// The mirror of Python's `clear_tape_cache()`: the hook a test uses to make a
/// following call a guaranteed miss, and the way to release the compiled pages
/// of a session that is done with a large system. Evaluators still in use by a
/// running call are kept alive by their `Arc`, so clearing mid-run is safe.
pub fn clear_cache() {
    let mut cache = lock();
    cache.entries.clear();
    cache.hits = 0;
    cache.misses = 0;
    cache.generation = cache.generation.wrapping_add(1);
}

/// Snapshot the cache counters.
pub fn cache_stats() -> JitCacheStats {
    let cache = lock();
    JitCacheStats {
        hits: cache.hits,
        misses: cache.misses,
        size: cache.entries.len(),
        maxsize: CACHE_MAXSIZE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex as StdMutex, MutexGuard as StdGuard, OnceLock};
    use tsdyn_ir::TapeBuilder;

    /// Test-only seam: run inside `cached_evaluator`'s compile window — after the
    /// compile, before the insert lock — so a test can land a `clear_cache()`
    /// exactly there without racing a sleep.
    static COMPILE_WINDOW_HOOK: StdMutex<Option<fn()>> = StdMutex::new(None);

    pub(super) fn run_compile_window_hook() {
        let hook = *COMPILE_WINDOW_HOOK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(f) = hook {
            f();
        }
    }

    fn set_compile_window_hook(f: Option<fn()>) {
        *COMPILE_WINDOW_HOOK
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = f;
    }

    #[test]
    fn a_clear_during_an_in_flight_compile_leaves_the_cache_empty() {
        let _g = serialize();
        clear_cache();
        set_compile_window_hook(Some(clear_cache));
        let ev = cached_evaluator(&scaled(7.0)).unwrap();
        set_compile_window_hook(None);
        // The caller still gets a usable evaluator...
        assert!(Arc::strong_count(&ev) >= 1);
        // ...but `clear_cache()` is a barrier: nothing was resurrected behind it.
        assert_eq!(cache_stats().size, 0);
        clear_cache();
    }

    /// The cache is process-wide, so the tests that assert on its counters must
    /// not interleave. `cargo test` runs them on threads by default.
    fn serialize() -> StdGuard<'static, ()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        let m = LOCK.get_or_init(|| StdMutex::new(()));
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `du/dt = [c * u0]` — one tape shape parameterised by the constant, so two
    /// calls differ **only** in `imm`.
    fn scaled(c: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let u0 = b.state(0);
        let k = b.constant(c);
        let r = b.mul(k, u0);
        b.finish(&[r], &[], 1, 0).unwrap()
    }

    /// The same registers and constants as `scaled(2.0)` but a different opcode
    /// *stream* (`Add` where `scaled` has `Mul`).
    fn added() -> Tape {
        let mut b = TapeBuilder::new();
        let u0 = b.state(0);
        let k = b.constant(2.0);
        let r = b.add(k, u0);
        b.finish(&[r], &[], 1, 0).unwrap()
    }

    /// `scaled(2.0)` plus the analytic Jacobian — same RHS instructions, the
    /// `with_jacobian` lowering's extra `jac_outputs`.
    fn scaled_with_jac() -> Tape {
        let mut b = TapeBuilder::new();
        let u0 = b.state(0);
        let k = b.constant(2.0);
        let r = b.mul(k, u0);
        b.finish(&[r], &[k], 1, 0).unwrap()
    }

    #[test]
    fn repeat_compile_is_a_hit() {
        let _g = serialize();
        clear_cache();
        let tape = scaled(2.0);
        let a = cached_evaluator(&tape).unwrap();
        assert_eq!(cache_stats().misses, 1);
        let b = cached_evaluator(&tape).unwrap();
        assert_eq!(cache_stats().hits, 1);
        assert_eq!(cache_stats().misses, 1);
        // Same compiled artifact, not merely an equal one.
        assert!(Arc::ptr_eq(&a, &b));
        clear_cache();
    }

    #[test]
    fn a_different_constant_is_a_miss() {
        let _g = serialize();
        clear_cache();
        cached_evaluator(&scaled(2.0)).unwrap();
        cached_evaluator(&scaled(3.0)).unwrap();
        let s = cache_stats();
        assert_eq!((s.hits, s.misses, s.size), (0, 2, 2));
        // …and each serves its own math.
        let two = cached_evaluator(&scaled(2.0)).unwrap();
        let three = cached_evaluator(&scaled(3.0)).unwrap();
        assert_eq!(two.eval_alloc(&[5.0], &[], 0.0), vec![10.0]);
        assert_eq!(three.eval_alloc(&[5.0], &[], 0.0), vec![15.0]);
        clear_cache();
    }

    #[test]
    fn a_different_opcode_sequence_is_a_miss() {
        let _g = serialize();
        clear_cache();
        cached_evaluator(&scaled(2.0)).unwrap();
        let other = cached_evaluator(&added()).unwrap();
        assert_eq!(cache_stats().misses, 2);
        assert_eq!(cache_stats().hits, 0);
        assert_eq!(other.eval_alloc(&[5.0], &[], 0.0), vec![7.0]);
        clear_cache();
    }

    #[test]
    fn a_different_with_jacobian_flag_is_a_miss() {
        let _g = serialize();
        clear_cache();
        let plain = cached_evaluator(&scaled(2.0)).unwrap();
        let with_jac = cached_evaluator(&scaled_with_jac()).unwrap();
        assert_eq!(cache_stats().misses, 2);
        assert!(!plain.has_jacobian());
        assert!(with_jac.has_jacobian());
        assert_eq!(with_jac.eval_jac_alloc(&[5.0], &[], 0.0).1, vec![2.0]);
        clear_cache();
    }

    /// The `imm` payload is part of `Tape`'s equality, so a tape differing only
    /// in a constant can never be served another's code. (Named by the module
    /// doc's hash note.)
    #[test]
    fn tape_eq_covers_imm() {
        assert_ne!(scaled(2.0), scaled(3.0));
        assert_ne!(tape_hash(&scaled(2.0)), tape_hash(&scaled(3.0)));
    }

    #[test]
    fn bypass_env_var_disables_the_cache() {
        let _g = serialize();
        clear_cache();
        let tape = scaled(2.0);
        std::env::set_var(CACHE_DISABLE_ENV, "1");
        let a = cached_evaluator(&tape).unwrap();
        let b = cached_evaluator(&tape).unwrap();
        std::env::remove_var(CACHE_DISABLE_ENV);
        // Distinct compiles, nothing stored, no counters touched.
        assert!(!Arc::ptr_eq(&a, &b));
        let s = cache_stats();
        assert_eq!((s.hits, s.misses, s.size), (0, 0, 0));
        // …and both give the same answers as a cached one.
        assert_eq!(
            a.eval_alloc(&[5.0], &[], 0.0),
            b.eval_alloc(&[5.0], &[], 0.0)
        );
        clear_cache();
    }

    #[test]
    fn eviction_bounds_the_store() {
        let _g = serialize();
        clear_cache();
        for i in 0..(CACHE_MAXSIZE + 5) {
            cached_evaluator(&scaled(i as f64)).unwrap();
        }
        let s = cache_stats();
        assert_eq!(s.size, CACHE_MAXSIZE);
        assert_eq!(s.maxsize, CACHE_MAXSIZE);
        // The most recent entries survived; the oldest were evicted.
        let newest = scaled((CACHE_MAXSIZE + 4) as f64);
        cached_evaluator(&newest).unwrap();
        assert_eq!(cache_stats().hits, 1);
        clear_cache();
    }

    #[test]
    fn concurrent_lookup_keeps_one_evaluator_per_tape() {
        let _g = serialize();
        clear_cache();
        let tape = scaled(7.0);
        let evs: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| s.spawn(|| cached_evaluator(&tape).unwrap()))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // Exactly one entry survives, and every thread's answer agrees.
        assert_eq!(cache_stats().size, 1);
        for ev in &evs {
            assert_eq!(ev.eval_alloc(&[2.0], &[], 0.0), vec![14.0]);
        }
        clear_cache();
    }
}
