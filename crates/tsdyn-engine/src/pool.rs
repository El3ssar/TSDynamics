//! The engine's rayon thread pool, rebuilt automatically after a `fork()`
//! (stream v6 WP2-safety).
//!
//! # The defect this exists to fix
//!
//! `fork()` duplicates only the calling thread. Every *other* thread in the
//! process — including a rayon worker — is gone in the child, but the pool's
//! bookkeeping (its job queues, its latches, the mutexes those threads happened
//! to hold at the instant of the fork) is copied over intact and now describes
//! workers that will never run again. The first parallel call in the child
//! therefore pushes a job onto a queue nobody is draining and blocks on a latch
//! nobody will set: it hangs, forever, inside the FFI, uninterruptibly.
//!
//! Because rayon's *global* pool is process-wide and built on first use, one
//! `ensemble()` anywhere in a session was enough to poison every subsequent
//! `multiprocessing` (fork-start-method) child that touched the engine. A
//! silent forever-hang is the worst failure mode a library can have — worse
//! than a crash, which at least says something — so it had to go.
//!
//! # The fix: a PID-tagged pool
//!
//! The engine owns its pool instead of borrowing the global one, and tags it
//! with the PID it was built under. [`with_pool`] compares that tag against the
//! live [`std::process::id`] on every call and rebuilds when they differ, so a
//! forked child transparently gets fresh workers on its first parallel call and
//! the parent keeps its own. This is the standard remedy (the same one
//! `numpy`/OpenBLAS and `tokio` users reach for) and the only one that leaves
//! `fork()` *working* rather than merely diagnosed: a `pthread_atfork` handler
//! cannot resurrect the lost threads, and detect-and-raise would break the
//! legitimate and common "fan work out with `multiprocessing`, integrate inside
//! each child" pattern.
//!
//! # Determinism is unaffected
//!
//! The pool only decides *which thread* runs a job. Every parallel loop in this
//! crate writes each trajectory into its own pre-assigned output row and seeds
//! it from its own index, so the result is bit-for-bit independent of thread
//! count and scheduling — the parallel == serial contract holds across a
//! rebuild exactly as it holds across a change in `RAYON_NUM_THREADS`.
//!
//! # The one case that stays broken (and always will)
//!
//! Forking *while an engine call is in flight on another thread* can copy a
//! locked [`Mutex`], and the child then deadlocks on [`POOL`] instead of on
//! rayon. That is unfixable in general — `fork()` in a multithreaded process is
//! only defined if the child goes straight to `exec()` — and it is not the case
//! anyone hits: `multiprocessing` forks from an idle parent. CPython itself
//! warns about the general hazard (`DeprecationWarning: This process is
//! multi-threaded, use of fork() may lead to deadlocks in the child`).

use std::sync::{Arc, Mutex, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// A built pool and the PID it belongs to. The pairing is the whole mechanism:
/// a pool is only reusable in the process that created its threads.
type PidTaggedPool = (u32, Arc<ThreadPool>);

/// The engine's pool, tagged with the PID it was built under.
///
/// `OnceLock<Mutex<..>>` rather than a `static Mutex` so the lock itself is
/// created lazily; the `Option` is `None` until the first parallel call.
static POOL: OnceLock<Mutex<Option<PidTaggedPool>>> = OnceLock::new();

/// Fetch the pool for the *current* process, building it if this is the first
/// parallel call here or if the cached one belongs to a pre-`fork` parent.
///
/// Returns `None` when the pool cannot be built (a thread-spawn failure under a
/// hard `RLIMIT_NPROC`, say); [`with_pool`] then falls back to running the work
/// on the calling thread, which is slower but always correct.
fn current_pool() -> Option<Arc<ThreadPool>> {
    let pid = std::process::id();
    let cell = POOL.get_or_init(|| Mutex::new(None));
    // A poisoned lock means a previous builder panicked; the cached pool is
    // still structurally fine to read, so recover rather than propagate.
    let mut slot = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((built_pid, pool)) = slot.as_ref() {
        if *built_pid == pid {
            return Some(Arc::clone(pool));
        }
        // Stale: inherited across a fork. Drop the handle without joining —
        // the threads it describes do not exist in this process, so any attempt
        // to wind them down would be the very hang we are avoiding.
        *slot = None;
    }
    let pool = Arc::new(ThreadPoolBuilder::new().build().ok()?);
    *slot = Some((pid, Arc::clone(&pool)));
    Some(pool)
}

/// Run `f` on the engine's (fork-safe) rayon pool.
///
/// Every parallel loop in this crate goes through here rather than calling
/// `par_iter()` on the ambient global pool, which is what makes the PID check
/// above unavoidable instead of merely available.
pub fn with_pool<R, F>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    match current_pool() {
        Some(pool) => pool.install(f),
        // No pool: run the (still parallel-shaped) work inline. Rayon's
        // iterators degrade to sequential execution on the calling thread, so
        // this is a slowdown, never a wrong answer.
        None => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn runs_work_and_reuses_one_pool_within_a_process() {
        let sum: i64 = with_pool(|| (0..1000i64).into_par_iter().sum());
        assert_eq!(sum, 499_500);

        let a = current_pool().expect("pool builds");
        let b = current_pool().expect("pool builds");
        assert!(
            Arc::ptr_eq(&a, &b),
            "the same process must reuse one pool, not rebuild per call"
        );
    }

    /// The PID tag is what makes a forked child rebuild. Simulate the stale
    /// entry directly (a real `fork` in a test harness is its own hazard):
    /// after planting a foreign PID, the next fetch must hand back a *different*
    /// pool.
    #[test]
    fn a_foreign_pid_tag_forces_a_rebuild() {
        let before = current_pool().expect("pool builds");
        {
            let cell = POOL.get().expect("initialised by the fetch above");
            let mut slot = cell.lock().unwrap_or_else(|e| e.into_inner());
            let (_, pool) = slot.take().expect("a pool was cached");
            // A PID that is not ours — exactly what a fork leaves behind.
            *slot = Some((std::process::id().wrapping_add(1), pool));
        }
        let after = current_pool().expect("pool rebuilds");
        assert!(
            !Arc::ptr_eq(&before, &after),
            "a pool tagged with another PID must be discarded, not reused"
        );
    }
}
