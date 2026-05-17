//! In-flight root finding on the integrator dense output (**M2 retrofit prep**).
//!
//! `detect_events` in Python remains on sampled-state Hermite until this module drives
//! a bracket + Brent refinement on native interpolants (small follow-up milestone after N5).
#![allow(dead_code)]

/// Placeholder for an event functor evaluated through the active method's interpolant.
#[derive(Debug, Clone)]
pub enum EventDriver {
    Deferred,
}
