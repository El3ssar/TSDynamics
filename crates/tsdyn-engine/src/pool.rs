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

use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::interrupt::{self, Cancel};

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

/// How long the driver waits for the batch before consulting the interrupt hook.
///
/// This bounds only how late a Ctrl-C is *noticed*, not how long the call takes:
/// the wait is a channel `recv_timeout`, so a batch that finishes wakes the
/// driver immediately and pays no part of this. 5 ms keeps the handshake
/// (which re-acquires the GIL) at 200/s — invisible next to a batch worth
/// parallelising — while staying two orders of magnitude inside the "a Ctrl-C
/// must land within a second or two" bar.
const HOOK_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Run `f` on the engine's pool **without parking the calling thread**, so a
/// Ctrl-C during a parallel batch is still seen.
///
/// # The defect this exists to fix
///
/// [`with_pool`] ends in `ThreadPool::install`, which runs the closure on a pool
/// *worker* and blocks the caller on a latch until it finishes. That is fatal
/// for interruption, because the calling thread is the only armed one (see
/// [`crate::interrupt`]): during an ensemble the one thread that could turn a
/// signal into a `KeyboardInterrupt` was asleep, and every worker skipped the
/// hook by design. So an ensemble ignored Ctrl-C completely — the exact
/// "mistyped `final_time` costs you the session" failure the interrupt work
/// existed to remove, on the calls most likely to run for minutes.
///
/// # The fix
///
/// Hand the fan-out to a scoped driver thread (which may park in `install` as
/// much as it likes) and keep the calling thread in a `recv_timeout` loop. It
/// wakes on completion, or every [`HOOK_POLL_INTERVAL`] to consult the hook; on
/// a stop it raises `cancel`, which the workers are already
/// [`watch`](crate::interrupt::watch)ing and poll on their normal stride. No
/// worker ever touches the GIL, and the batch's positional partition — the
/// parallel == serial contract — is untouched: this changes *which thread waits*,
/// not how the work is divided.
///
/// The driver thread is only spawned when the caller is armed, i.e. when there
/// is an embedder that could raise something; an unarmed caller (every in-crate
/// Rust test, any non-Python embedder) takes [`with_pool`] unchanged and pays
/// nothing. If the driver thread cannot be spawned at all, the work runs inline
/// — the same graceful degradation [`with_pool`] already does when the rayon
/// pool itself cannot be built.
pub fn with_pool_interruptible<R, F>(cancel: &Cancel, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    // Nothing here can turn a cancellation into an exception, so a driver thread
    // would be pure overhead.
    if !interrupt::is_armed() {
        return with_pool(f);
    }
    let (tx, rx) = mpsc::channel::<()>();
    // `Mutex<Option<F>>` rather than moving `f` straight into the closure, so the
    // job can be reclaimed and run inline should the spawn fail. Declared outside
    // the scope because the driver closure borrows it for all of `'scope`.
    let job = Mutex::new(Some(f));
    // A `&Mutex` (Copy) so the closure below can be `move` — which it MUST be, to
    // take ownership of `tx`. `Sender::send` needs only `&self`, so a non-`move`
    // closure would capture the sender by *reference* and leave it alive on this
    // frame: the receiver would then never see `Disconnected`, and a panicking
    // batch (which never reaches the `send`) would leave this thread polling for
    // a completion that can no longer arrive. `move` alone is not enough either —
    // it would swallow `job`, which the spawn-failure path still needs.
    let job_ref = &job;
    thread::scope(|scope| {
        let spawned = thread::Builder::new()
            .name("tsdyn-batch".to_string())
            .spawn_scoped(scope, move || {
                let out = with_pool(take_job(job_ref));
                // Wake the caller the instant the batch is done, so a short batch
                // pays none of HOOK_POLL_INTERVAL. A panic in `f` skips this and
                // drops the sender instead, which the loop reads as
                // `Disconnected` — it stops waiting either way.
                let _ = tx.send(());
                out
            });
        let Ok(handle) = spawned else {
            return with_pool(take_job(job_ref));
        };
        loop {
            match rx.recv_timeout(HOOK_POLL_INTERVAL) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                Err(RecvTimeoutError::Timeout) => {
                    if interrupt::poll_hook() {
                        cancel.cancel();
                    }
                }
            }
        }
        match handle.join() {
            Ok(out) => out,
            // Carry a worker panic across the thread boundary rather than
            // swallowing it into a `thread::scope` abort at the end of the block.
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

/// Take the one-shot job out of its slot (it is claimed exactly once, by
/// whichever of the driver thread or the fallback path gets there).
fn take_job<F>(job: &Mutex<Option<F>>) -> F {
    job.lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .expect("the batch job is claimed exactly once")
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

    /// The headline property: the calling thread stays awake and *cancels*.
    ///
    /// This is the regression test for the parked-caller defect. The batch stands
    /// in for a long fan-out by waiting for the flag; under the old
    /// `install`-and-park the calling thread would have been asleep for the whole
    /// wait, nothing would ever have raised it, and the closure would return
    /// `false` at its deadline (a *failure*, deliberately, rather than a hang).
    #[test]
    fn the_calling_thread_polls_and_cancels_while_the_batch_runs() {
        let _stop = interrupt::testing::force_stop();
        let _armed = interrupt::arm();

        let cancel = Cancel::new();
        let observed = with_pool_interruptible(&cancel, || {
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !cancel.is_cancelled() && std::time::Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            cancel.is_cancelled()
        });

        assert!(
            observed,
            "the armed calling thread never raised the cancellation flag — it is \
             parked inside the pool again"
        );
        assert!(cancel.is_cancelled());
    }

    /// An unarmed caller takes the old path exactly: same answer, no flag, and
    /// (the point) no driver thread to pay for.
    #[test]
    fn an_unarmed_caller_runs_the_batch_unchanged() {
        assert!(!interrupt::is_armed());
        let cancel = Cancel::new();
        let sum: i64 = with_pool_interruptible(&cancel, || (0..1000i64).into_par_iter().sum());
        assert_eq!(sum, 499_500);
        assert!(
            !cancel.is_cancelled(),
            "nothing should cancel a quiet batch"
        );
    }

    /// A hook that never stops must leave the batch alone — the poll loop is a
    /// watchdog, not a deadline.
    #[test]
    fn a_quiet_hook_does_not_disturb_an_armed_batch() {
        interrupt::testing::install_hook();
        let _armed = interrupt::arm();
        let cancel = Cancel::new();
        // Long enough to cross HOOK_POLL_INTERVAL many times over.
        let sum: i64 = with_pool_interruptible(&cancel, || {
            thread::sleep(Duration::from_millis(60));
            (0..1000i64).into_par_iter().sum()
        });
        assert_eq!(sum, 499_500);
        assert!(!cancel.is_cancelled());
    }

    /// A panic inside the batch must reach the caller, not be swallowed by the
    /// driver thread (or turned into a `thread::scope` abort).
    #[test]
    #[should_panic(expected = "a trajectory panicked")]
    fn a_panic_in_the_batch_propagates_to_the_caller() {
        let _armed = interrupt::arm();
        let cancel = Cancel::new();
        with_pool_interruptible(&cancel, || panic!("a trajectory panicked"));
    }
}
