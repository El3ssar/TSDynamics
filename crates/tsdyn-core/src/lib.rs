//! `tsdyn-core` — the PyO3 binding layer that builds the `tsdynamics._rust`
//! extension module.
//!
//! Thin by design: marshal numpy arrays to/from `&[f64]` slices (zero-copy),
//! release the GIL around the Rust work, and dispatch to `tsdyn-engine`. This is
//! the only crate that knows about Python; everything below it is pure Rust.
//!
//! Skeleton only (stream F0). The bindings — and the `[lib] crate-type =
//! ["cdylib"]` / pyo3 setup, plus the Python-side `tsdynamics/_rust` package —
//! land in **stream E7**. See ROADMAP §4a/§4b.
