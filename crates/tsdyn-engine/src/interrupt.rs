//! Cooperative cancellation, so a long engine call can be interrupted
//! (stream v6 WP2-safety).
//!
//! # The defect this exists to fix
//!
//! Ctrl-C sets a flag; Python only turns that flag into a `KeyboardInterrupt`
//! when the *interpreter* next runs bytecode. A call that spends four minutes
//! inside Rust runs no bytecode for four minutes, so the interrupt is simply
//! deferred until the FFI call returns. For an interactive scientific library
//! that is a serious usability defect: one mistyped `final_time` and the only
//! way out of the session is to kill it — losing every unsaved variable in the
//! notebook.
//!
//! # The design
//!
//! The engine crate is deliberately Python-free, so it cannot call
//! `PyErr_CheckSignals` itself. Instead the embedder (the `tsdyn-core` binding
//! crate) [`install`]s a hook once, and the long sequential loops poll it
//! through a [`Poller`]. When the hook says "stop", the loop unwinds through
//! its family's `Interrupted` error variant and the binding layer re-raises
//! whatever the signal handler produced — normally `KeyboardInterrupt`.
//!
//! # Two properties the implementation has to have
//!
//! **It must cost nothing.** [`Poller::tick`] is an increment and a compare
//! against [`POLL_STRIDE`]; the hook — which needs to re-acquire the GIL, and
//! is therefore the expensive part — runs on one iteration in `POLL_STRIDE`.
//! At a stride of 4096 the amortised cost is far below the noise floor of a
//! single RHS evaluation.
//!
//! **It must only fire on the calling thread.** `PyErr_CheckSignals` is a no-op
//! off the main thread, and re-acquiring the GIL from each of N rayon workers
//! would serialise a parallel loop on the one lock the engine works hard to
//! stay out of. So polling is *armed per thread*: the binding layer calls
//! [`arm`] inside the closure it hands to `Python::detach`, which runs on the
//! Python thread that made the call, and worker threads — never armed — skip
//! the hook entirely for the price of one thread-local read per stride.
//!
//! # Parallel loops: the [`Cancel`] flag
//!
//! Those two properties together left one hole, and it was a big one: the
//! *ensembles*. A rayon fan-out runs every trajectory on a worker, and workers
//! are deliberately never armed — so no thread inside an ensemble ever consulted
//! the hook, and Ctrl-C during a thousand-trajectory batch did nothing at all.
//!
//! [`Cancel`] closes it without giving up either property. It is a plain shared
//! `AtomicBool` that a worker [`watch`]es for the life of one trajectory and
//! reads — **relaxed, no GIL, no shared lock** — on the same stride it would
//! otherwise have consulted the hook on. Nothing sets it but the *calling*
//! thread, which stays armed, stays unparked (see
//! [`crate::pool::with_pool_interruptible`]) and keeps polling signals while the
//! batch runs. The signal handshake therefore still happens on exactly one
//! thread; all the workers ever do is read a bool.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// How many loop iterations pass between two consultations of the hook.
///
/// Chosen so the *upper bound* on interrupt latency stays interactive even for
/// the cheapest possible iteration (a 1-D map step, a few nanoseconds: 4096 of
/// them is microseconds) while the per-iteration cost stays a single predictable
/// branch. There is no benefit to a smaller stride — the human at the keyboard
/// cannot tell 10 µs from 10 ms — and a larger one starts to matter for a truly
/// expensive single step.
///
/// The stride counts *units of engine work* — one solver step, one map iterate —
/// never segments. That is why the callers that chop a run into many short
/// segments (the Lyapunov renormalisation chunks, the basin cell march) thread
/// **one** poller down into the step loop instead of creating one per segment: a
/// per-segment poller would reset before it ever reached a stride, and the
/// longest-running calls in the library would never poll at all.
pub const POLL_STRIDE: usize = 4096;

/// The embedder's "should this call stop now?" predicate.
///
/// A plain `fn` pointer rather than a boxed closure: it is installed once, read
/// on a hot-ish path, and needs no captured state (the binding layer keeps the
/// interrupting error in its own thread-local).
pub type Hook = fn() -> bool;

static HOOK: OnceLock<Hook> = OnceLock::new();

thread_local! {
    /// Whether *this* thread should consult the hook. See the module docs.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

/// Install the process-wide interrupt hook.
///
/// Idempotent and first-wins: a second call is ignored, so a library embedding
/// the engine twice cannot silently displace the first embedder's handler.
pub fn install(hook: Hook) {
    let _ = HOOK.set(hook);
}

/// Whether a hook has been installed (the engine runs uninterruptible without
/// one, exactly as it did before this module existed).
pub fn is_installed() -> bool {
    HOOK.get().is_some()
}

/// Arms interrupt polling on the current thread until it is dropped.
///
/// Restores the previous state on drop rather than clearing unconditionally, so
/// nesting (an armed entry point calling another) is safe.
#[derive(Debug)]
pub struct ArmGuard {
    previous: bool,
}

impl Drop for ArmGuard {
    fn drop(&mut self) {
        ARMED.with(|a| a.set(self.previous));
    }
}

/// Arm interrupt polling on the calling thread for the returned guard's
/// lifetime.
///
/// Call this on the thread that owns the Python call — inside the
/// `Python::detach` closure — never on a worker.
#[must_use = "polling is armed only while the guard is alive"]
pub fn arm() -> ArmGuard {
    ArmGuard {
        previous: ARMED.with(|a| a.replace(true)),
    }
}

/// Whether the calling thread is currently armed (test/diagnostic helper).
pub fn is_armed() -> bool {
    ARMED.with(Cell::get)
}

/// A cancellation flag shared by one parallel batch and its workers.
///
/// The GIL-free half of the interrupt story (see the [module docs](self)): the
/// armed calling thread raises it, and every worker [`watch`]ing it stops at its
/// next [`Poller::tick`] stride. Cloning is an `Arc` bump, so handing one to each
/// of a thousand trajectories costs nothing measurable.
///
/// [`Ordering::Relaxed`] is the right and sufficient ordering: the flag publishes
/// no data of its own — the batch's results are handed back across a thread
/// `join`, which is a full happens-before edge — so all it has to be is
/// *eventually visible*, which relaxed atomics guarantee.
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A fresh, un-raised flag.
    pub fn new() -> Self {
        Cancel::default()
    }

    /// Ask every thread watching this flag to stop at its next poll.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

thread_local! {
    /// The cancellation flag *this* thread is currently working under, if any.
    static WATCHED: RefCell<Option<Cancel>> = const { RefCell::new(None) };
}

/// Watches a [`Cancel`] on the current thread until it is dropped.
///
/// Restores the previous flag rather than clearing, so nesting is safe.
#[derive(Debug)]
pub struct WatchGuard {
    previous: Option<Cancel>,
}

impl Drop for WatchGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        WATCHED.with(|w| *w.borrow_mut() = previous);
    }
}

/// Make the calling thread stop when `flag` is raised, for the guard's lifetime.
///
/// Call this on a *worker* at the top of the unit of work it should be able to
/// abandon — one ensemble trajectory. The cost is one `Arc` clone per unit of
/// work, never per step.
#[must_use = "the flag is watched only while the guard is alive"]
pub fn watch(flag: &Cancel) -> WatchGuard {
    WatchGuard {
        previous: WATCHED.with(|w| w.borrow_mut().replace(flag.clone())),
    }
}

/// A strided interrupt poll, one per long-running loop.
///
/// Keep one in the loop's frame and call [`tick`](Poller::tick) once per
/// iteration; act on `true` by returning the family's `Interrupted` error.
#[derive(Clone, Copy, Debug, Default)]
pub struct Poller {
    since_last: usize,
}

impl Poller {
    /// A fresh poller (fires no earlier than [`POLL_STRIDE`] ticks from now).
    pub fn new() -> Self {
        Poller::default()
    }

    /// Count one loop iteration; return `true` when the caller should stop.
    ///
    /// The common path is an increment and a predictable not-taken branch — the
    /// actual check is `#[inline(never)]` + `#[cold]` so its GIL handshake never
    /// bloats the loop body it guards.
    #[inline]
    pub fn tick(&mut self) -> bool {
        self.since_last += 1;
        if self.since_last < POLL_STRIDE {
            return false;
        }
        self.since_last = 0;
        check()
    }
}

/// The strided stop check: a watched cancellation flag, then the hook.
///
/// The flag is tested first and unconditionally, because it is the *worker's*
/// only channel — a worker is never armed, so [`poll_hook`] is a single
/// thread-local read that always returns `false` there.
#[cold]
#[inline(never)]
fn check() -> bool {
    if WATCHED.with(|w| w.borrow().as_ref().is_some_and(Cancel::is_cancelled)) {
        return true;
    }
    poll_hook()
}

/// Consult the embedder's hook, if this thread is armed and a hook exists.
///
/// Public because the ensemble driver ([`crate::pool::with_pool_interruptible`])
/// polls it directly from its wait loop rather than through a [`Poller`]: it is
/// running no numerical loop of its own to count strides against, so it polls on
/// a wall-clock cadence instead.
pub fn poll_hook() -> bool {
    if !ARMED.with(Cell::get) {
        return false;
    }
    match HOOK.get() {
        Some(hook) => hook(),
        None => false,
    }
}

/// The one interrupt hook the crate's tests use.
///
/// [`install`] is process-wide and first-wins, and a `cargo test` binary runs
/// every module's tests in *one* process — so there can only be one test hook,
/// and every test that wants to drive an interrupt has to drive this one. Its
/// state is **thread-local**, which is what keeps tests independent: the test
/// harness gives each test its own thread, so one test forcing an interrupt
/// cannot leak into another running concurrently.
#[cfg(test)]
pub(crate) mod testing {
    use std::cell::Cell;

    thread_local! {
        static STOP: Cell<bool> = const { Cell::new(false) };
        static CALLS: Cell<usize> = const { Cell::new(0) };
    }

    fn hook() -> bool {
        CALLS.with(|c| c.set(c.get() + 1));
        STOP.with(Cell::get)
    }

    /// Install the shared test hook. Idempotent, and safe to call from every
    /// test: they all pass the same function pointer, so first-wins is a no-op.
    pub(crate) fn install_hook() {
        super::install(hook);
    }

    /// How many times this thread has consulted the hook.
    pub(crate) fn calls() -> usize {
        CALLS.with(Cell::get)
    }

    /// Make the hook say "stop" on this thread until the guard drops.
    #[must_use]
    pub(crate) fn force_stop() -> StopGuard {
        install_hook();
        StopGuard {
            previous: STOP.with(|s| s.replace(true)),
        }
    }

    /// Restores the previous "stop" state on drop.
    pub(crate) struct StopGuard {
        previous: bool,
    }

    impl Drop for StopGuard {
        fn drop(&mut self) {
            STOP.with(|s| s.set(self.previous));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One consultation per `POLL_STRIDE` ticks — not one per tick — and only
    /// on an armed thread.
    #[test]
    fn polls_on_the_stride_and_only_when_armed() {
        testing::install_hook();
        assert!(is_installed());

        // Unarmed: the hook is never consulted, however many ticks pass.
        let before = testing::calls();
        let mut p = Poller::new();
        for _ in 0..(4 * POLL_STRIDE) {
            assert!(!p.tick());
        }
        assert_eq!(
            testing::calls(),
            before,
            "an unarmed thread must not reach the hook"
        );

        // Armed: exactly one consultation per stride.
        let guard = arm();
        assert!(is_armed());
        let before = testing::calls();
        let mut p = Poller::new();
        for _ in 0..(3 * POLL_STRIDE) {
            assert!(!p.tick(), "a hook returning false never stops the loop");
        }
        assert_eq!(testing::calls(), before + 3);
        drop(guard);
        assert!(!is_armed(), "the guard disarms on drop");
    }

    /// A hook that says "stop" stops the loop — and only on the armed thread.
    #[test]
    fn a_stopping_hook_is_reported_only_when_armed() {
        let _stop = testing::force_stop();

        let mut p = Poller::new();
        for _ in 0..(2 * POLL_STRIDE) {
            assert!(!p.tick(), "unarmed threads ignore the hook entirely");
        }

        let _armed = arm();
        let mut p = Poller::new();
        let mut stopped = false;
        for _ in 0..POLL_STRIDE {
            if p.tick() {
                stopped = true;
                break;
            }
        }
        assert!(
            stopped,
            "an armed thread must observe the stop within a stride"
        );
    }

    /// Arming nests: an inner guard must not disarm the outer scope.
    #[test]
    fn arming_nests() {
        let outer = arm();
        {
            let _inner = arm();
            assert!(is_armed());
        }
        assert!(
            is_armed(),
            "the inner guard restored 'armed', not 'unarmed'"
        );
        drop(outer);
        assert!(!is_armed());
    }
}
