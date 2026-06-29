//! The [`Tape`] — a compiled right-hand side in flat, single-static-assignment
//! form, plus the [`IrError`] raised when a tape is malformed.
//!
//! # Layout
//!
//! A tape is a list of `n_reg` instructions stored as **parallel arrays** of
//! equal length (`ops`, `a`, `b`, `imm`).  Instruction `i` writes register `i`
//! (single static assignment); later instructions read earlier registers by
//! index.  Common subexpressions are shared at build time, so the tape is
//! already CSE'd.  Evaluating it is one linear pass over the arrays with a
//! reusable register scratch buffer — allocation-free and free of any callback
//! into Python.
//!
//! What each instruction reads, by [`OpKind`](crate::OpKind):
//!
//! | Field    | `Leaf`                              | `Unary`        | `Binary`             | `Powi`              |
//! |----------|-------------------------------------|----------------|----------------------|---------------------|
//! | `ops[i]` | the opcode                          | the opcode     | the opcode           | `Op::Powi`          |
//! | `a[i]`   | `State`/`Param`: input index; else — | source reg     | left source reg      | base source reg     |
//! | `b[i]`   | —                                   | —              | right source reg     | **integer exponent**|
//! | `imm[i]` | `Const`: the constant; else —       | —              | —                    | —                   |
//!
//! `imm` is parallel to `ops` (one slot per instruction) and only read for
//! `Const`; the other slots are conventionally `0.0`.  This "immediate at the
//! instruction's own index" layout is the v2 contract and is preserved verbatim.
//!
//! # Outputs
//!
//! `outputs[k]` is the register holding derivative component `k`; its length is
//! the system dimension.  `jac_outputs` is empty, or holds the registers of the
//! row-major `dim × dim` Jacobian `∂f_k/∂u_j` (`jac_outputs[k*dim + j]`) for the
//! kernels that need it (the stiff/implicit family).
//!
//! `n_state` and `n_param` are the declared input widths; they bound the leaf
//! indices and are validated, but the evaluator does not read them (indices are
//! baked into the instructions).

use crate::op::{Op, OpKind};

/// Why a sequence of arrays is not a well-formed [`Tape`].
///
/// The v2 VM left these unchecked — it indexed blindly and returned `NaN` for an
/// unknown opcode.  As the frozen contract, the IR validates at construction so
/// a malformed tape fails at the FFI boundary instead of corrupting a hot loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrError {
    /// `ops`, `a`, `b`, `imm` are not all the same length.
    LengthMismatch {
        /// `ops.len()` — the canonical instruction count.
        ops: usize,
        /// `a.len()`.
        a: usize,
        /// `b.len()`.
        b: usize,
        /// `imm.len()`.
        imm: usize,
    },
    /// An `ops` entry is not a known opcode (the offending wire value).
    UnknownOpcode(i32),
    /// Instruction `at` reads register `reg`, which is not a strictly earlier
    /// instruction (`0 <= reg < at` is required by single static assignment).
    ForwardReference {
        /// The reading instruction's index.
        at: usize,
        /// The out-of-order register index it tried to read.
        reg: i32,
    },
    /// A `State` leaf at instruction `at` indexes `idx`, outside `0..n_state`.
    StateIndexOutOfRange {
        /// The leaf instruction's index.
        at: usize,
        /// The offending state index.
        idx: i32,
        /// The declared state width.
        n_state: usize,
    },
    /// A `Param` leaf at instruction `at` indexes `idx`, outside `0..n_param`.
    ParamIndexOutOfRange {
        /// The leaf instruction's index.
        at: usize,
        /// The offending parameter index.
        idx: i32,
        /// The declared parameter width.
        n_param: usize,
    },
    /// An `outputs[k]` register index is outside `0..n_reg`.
    OutputIndexOutOfRange {
        /// Position within `outputs`.
        k: usize,
        /// The offending register index.
        reg: i32,
        /// The instruction count.
        n_reg: usize,
    },
    /// A `jac_outputs` register index is outside `0..n_reg`.
    JacIndexOutOfRange {
        /// Position within `jac_outputs`.
        k: usize,
        /// The offending register index.
        reg: i32,
        /// The instruction count.
        n_reg: usize,
    },
    /// `jac_outputs` is non-empty but its length is not `dim * dim`.
    JacShapeMismatch {
        /// `jac_outputs.len()`.
        len: usize,
        /// The system dimension (`outputs.len()`).
        dim: usize,
    },
}

impl core::fmt::Display for IrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IrError::LengthMismatch { ops, a, b, imm } => write!(
                f,
                "tape arrays have mismatched lengths: ops={ops}, a={a}, b={b}, imm={imm}"
            ),
            IrError::UnknownOpcode(v) => write!(f, "unknown opcode {v}"),
            IrError::ForwardReference { at, reg } => write!(
                f,
                "instruction {at} reads register {reg}, which is not a strictly earlier register"
            ),
            IrError::StateIndexOutOfRange { at, idx, n_state } => write!(
                f,
                "instruction {at}: state index {idx} out of range for n_state={n_state}"
            ),
            IrError::ParamIndexOutOfRange { at, idx, n_param } => write!(
                f,
                "instruction {at}: param index {idx} out of range for n_param={n_param}"
            ),
            IrError::OutputIndexOutOfRange { k, reg, n_reg } => {
                write!(f, "outputs[{k}] = {reg} is out of range for n_reg={n_reg}")
            }
            IrError::JacIndexOutOfRange { k, reg, n_reg } => write!(
                f,
                "jac_outputs[{k}] = {reg} is out of range for n_reg={n_reg}"
            ),
            IrError::JacShapeMismatch { len, dim } => {
                write!(f, "jac_outputs length {len} is not dim*dim = {dim}*{dim}")
            }
        }
    }
}

impl std::error::Error for IrError {}

/// A compiled right-hand side `du/dt = f(u, p, t)` (and optionally its
/// Jacobian) as a flat instruction tape.
///
/// Construct one with [`TapeBuilder`](crate::TapeBuilder) (typed, in-process) or
/// [`Tape::from_arrays`] (the FFI path from the Python emitter).  Both validate;
/// a `Tape` value is therefore always well-formed.  Fields are private — read
/// them through the slice accessors, which downstream evaluators ([`tsdyn-vm`],
/// [`tsdyn-jit`]) use in their hot loops at zero cost.
///
/// [`tsdyn-vm`]: https://docs.rs/tsdyn-vm
/// [`tsdyn-jit`]: https://docs.rs/tsdyn-jit
#[derive(Clone, Debug, PartialEq)]
pub struct Tape {
    ops: Vec<Op>,
    a: Vec<i32>,
    b: Vec<i32>,
    imm: Vec<f64>,
    outputs: Vec<i32>,
    jac_outputs: Vec<i32>,
    n_state: usize,
    n_param: usize,
}

impl Tape {
    /// Build a tape from decoded parts, validating well-formedness.
    ///
    /// Prefer [`TapeBuilder`](crate::TapeBuilder) for hand-written tapes and
    /// [`Tape::from_arrays`] for the raw-integer FFI path; this is the shared
    /// validating constructor they both funnel through.
    // The eight parallel arrays *are* the tape contract; bundling them into a
    // struct just to pass them here would add a type without removing a field.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        ops: Vec<Op>,
        a: Vec<i32>,
        b: Vec<i32>,
        imm: Vec<f64>,
        outputs: Vec<i32>,
        jac_outputs: Vec<i32>,
        n_state: usize,
        n_param: usize,
    ) -> Result<Tape, IrError> {
        let tape = Tape {
            ops,
            a,
            b,
            imm,
            outputs,
            jac_outputs,
            n_state,
            n_param,
        };
        tape.validate()?;
        Ok(tape)
    }

    /// Build a tape from the raw wire arrays — the FFI ingestion path.
    ///
    /// `ops` are decoded to [`Op`]s (rejecting unknown opcodes); the remaining
    /// arrays are validated by [`from_parts`](Tape::from_parts).  This is what
    /// the PyO3 layer calls with zero-copy NumPy slices.
    #[allow(clippy::too_many_arguments)] // mirrors `from_parts`: the arrays are the contract.
    pub fn from_arrays(
        ops: &[i32],
        a: &[i32],
        b: &[i32],
        imm: &[f64],
        outputs: &[i32],
        jac_outputs: &[i32],
        n_state: usize,
        n_param: usize,
    ) -> Result<Tape, IrError> {
        let decoded: Result<Vec<Op>, IrError> = ops.iter().map(|&v| Op::from_i32(v)).collect();
        Tape::from_parts(
            decoded?,
            a.to_vec(),
            b.to_vec(),
            imm.to_vec(),
            outputs.to_vec(),
            jac_outputs.to_vec(),
            n_state,
            n_param,
        )
    }

    /// Check every structural invariant.  Called by the constructors; exposed so
    /// a caller that mutates a tape through some future API can re-check it.
    pub fn validate(&self) -> Result<(), IrError> {
        let n = self.ops.len();
        if self.a.len() != n || self.b.len() != n || self.imm.len() != n {
            return Err(IrError::LengthMismatch {
                ops: n,
                a: self.a.len(),
                b: self.b.len(),
                imm: self.imm.len(),
            });
        }

        for i in 0..n {
            let a = self.a[i];
            let b = self.b[i];
            match self.ops[i].kind() {
                OpKind::Leaf => match self.ops[i] {
                    Op::State if a < 0 || a as usize >= self.n_state => {
                        return Err(IrError::StateIndexOutOfRange {
                            at: i,
                            idx: a,
                            n_state: self.n_state,
                        });
                    }
                    Op::Param if a < 0 || a as usize >= self.n_param => {
                        return Err(IrError::ParamIndexOutOfRange {
                            at: i,
                            idx: a,
                            n_param: self.n_param,
                        });
                    }
                    // In-range State/Param, plus Const (reads imm[i]) and Time
                    // (reads nothing): no operand bounds to check.
                    _ => {}
                },
                OpKind::Unary => check_reg(i, a)?,
                OpKind::Binary => {
                    check_reg(i, a)?;
                    check_reg(i, b)?;
                }
                // Powi reads register `a`; `b` is the literal exponent.
                OpKind::Powi => check_reg(i, a)?,
            }
        }

        for (k, &reg) in self.outputs.iter().enumerate() {
            if reg < 0 || reg as usize >= n {
                return Err(IrError::OutputIndexOutOfRange { k, reg, n_reg: n });
            }
        }

        let dim = self.outputs.len();
        if !self.jac_outputs.is_empty() && self.jac_outputs.len() != dim * dim {
            return Err(IrError::JacShapeMismatch {
                len: self.jac_outputs.len(),
                dim,
            });
        }
        for (k, &reg) in self.jac_outputs.iter().enumerate() {
            if reg < 0 || reg as usize >= n {
                return Err(IrError::JacIndexOutOfRange { k, reg, n_reg: n });
            }
        }

        Ok(())
    }

    /// Number of instructions (= number of registers).
    #[inline]
    pub fn n_reg(&self) -> usize {
        self.ops.len()
    }

    /// System dimension (number of derivative outputs).
    #[inline]
    pub fn dim(&self) -> usize {
        self.outputs.len()
    }

    /// Declared state width.
    #[inline]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Declared parameter width.
    #[inline]
    pub fn n_param(&self) -> usize {
        self.n_param
    }

    /// Whether this tape carries a Jacobian (`jac_outputs` populated).
    #[inline]
    pub fn has_jacobian(&self) -> bool {
        !self.jac_outputs.is_empty()
    }

    /// The decoded opcodes.
    #[inline]
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// The `a` operands (input index for leaves, else a source register).
    #[inline]
    pub fn a(&self) -> &[i32] {
        &self.a
    }

    /// The `b` operands (right source register for binaries, integer exponent
    /// for `Powi`).
    #[inline]
    pub fn b(&self) -> &[i32] {
        &self.b
    }

    /// The immediates (read only by `Const`, at the instruction's own index).
    #[inline]
    pub fn imm(&self) -> &[f64] {
        &self.imm
    }

    /// Registers holding each derivative output.
    #[inline]
    pub fn outputs(&self) -> &[i32] {
        &self.outputs
    }

    /// Registers holding the row-major `dim × dim` Jacobian (empty if none).
    #[inline]
    pub fn jac_outputs(&self) -> &[i32] {
        &self.jac_outputs
    }

    /// Re-encode `ops` to the wire integers (the inverse of `from_arrays`'
    /// decode), for serialization and round-trip checks.
    pub fn ops_i32(&self) -> Vec<i32> {
        self.ops.iter().map(|op| op.to_i32()).collect()
    }

    /// Liveness / reachability mask over the registers: `mask[i]` is `true` iff
    /// some seed output transitively depends on register `i`.
    ///
    /// The seed set is the derivative [`outputs`](Tape::outputs), plus the
    /// [`jac_outputs`](Tape::jac_outputs) when `include_jac` is `true`. This is
    /// the single, shared dead-register-elimination (DCE) backward pass both
    /// evaluators call:
    ///
    /// - `tsdyn-vm`'s `Interpreter` passes `include_jac = false` to build its
    ///   RHS-only execution mask (skip Jacobian-only subexpressions in `eval`).
    /// - `tsdyn-jit`'s code generator calls it with `include_jac = false` for the
    ///   `tsdyn_eval` body and `include_jac = true` for the `tsdyn_eval_jac` body
    ///   (emit only the registers each function's outputs reach).
    ///
    /// Hoisting it here makes the invariant *the JIT-emitted register set equals
    /// the interpreter-executed register set* structural rather than maintained in
    /// two parallel copies — both backends now read the same mask, so they cannot
    /// drift apart by a register.
    ///
    /// # Algorithm
    ///
    /// One backward sweep. The tape is strict single static assignment — every
    /// operand index is strictly less than the register the op writes — so
    /// visiting registers in reverse and propagating need from a live op to its
    /// operands marks the full transitive dependency set in a single pass (an
    /// operand register is always visited *after* every op that reads it). The
    /// operand structure is read from [`Op::kind`](crate::Op): a `Leaf` reads
    /// nothing, `Unary`/`Powi` read `a`, and `Binary` reads `a` and `b`.
    ///
    /// The returned mask has length [`n_reg`](Tape::n_reg).
    pub fn reachable_from(&self, include_jac: bool) -> Vec<bool> {
        let n = self.ops.len();
        let mut live = vec![false; n];

        // Seed: every derivative-output register is live. Jacobian-output
        // registers are seeded only when the caller wants the Jacobian path.
        for &r in &self.outputs {
            live[r as usize] = true;
        }
        if include_jac {
            for &r in &self.jac_outputs {
                live[r as usize] = true;
            }
        }

        for i in (0..n).rev() {
            if !live[i] {
                continue;
            }
            match self.ops[i].kind() {
                OpKind::Leaf => {}
                OpKind::Unary | OpKind::Powi => live[self.a[i] as usize] = true,
                OpKind::Binary => {
                    live[self.a[i] as usize] = true;
                    live[self.b[i] as usize] = true;
                }
            }
        }
        live
    }
}

/// A source register must be a strictly earlier instruction.
#[inline]
fn check_reg(at: usize, reg: i32) -> Result<(), IrError> {
    if reg < 0 || reg as usize >= at {
        Err(IrError::ForwardReference { at, reg })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal valid tape: f(u) = u0 * p0  (dim 1, n_state 1, n_param 1).
    #[allow(clippy::type_complexity)] // a one-off test fixture tuple; a named type would not aid clarity.
    fn ok_parts() -> (Vec<Op>, Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>) {
        let ops = vec![Op::State, Op::Param, Op::Mul];
        let a = vec![0, 0, 0];
        let b = vec![0, 0, 1];
        let imm = vec![0.0, 0.0, 0.0];
        let outputs = vec![2];
        (ops, a, b, imm, outputs)
    }

    #[test]
    fn valid_tape_constructs() {
        let (ops, a, b, imm, outputs) = ok_parts();
        let t = Tape::from_parts(ops, a, b, imm, outputs, vec![], 1, 1).unwrap();
        assert_eq!(t.n_reg(), 3);
        assert_eq!(t.dim(), 1);
        assert!(!t.has_jacobian());
        assert_eq!(t.ops_i32(), vec![1, 2, 12]);
    }

    #[test]
    fn length_mismatch_is_rejected() {
        let (ops, mut a, b, imm, outputs) = ok_parts();
        a.push(0); // now len 4 vs ops len 3
        let err = Tape::from_parts(ops, a, b, imm, outputs, vec![], 1, 1).unwrap_err();
        assert!(matches!(err, IrError::LengthMismatch { ops: 3, a: 4, .. }));
    }

    #[test]
    fn forward_reference_is_rejected() {
        // Mul at index 2 reads register b=3, which does not exist yet.
        let (ops, a, mut b, imm, outputs) = ok_parts();
        b[2] = 3;
        let err = Tape::from_parts(ops, a, b, imm, outputs, vec![], 1, 1).unwrap_err();
        assert!(matches!(err, IrError::ForwardReference { at: 2, reg: 3 }));
    }

    #[test]
    fn self_reference_is_a_forward_reference() {
        // Mul at index 2 reading register a=2 (itself) is not strictly earlier.
        let (ops, mut a, b, imm, outputs) = ok_parts();
        a[2] = 2;
        let err = Tape::from_parts(ops, a, b, imm, outputs, vec![], 1, 1).unwrap_err();
        assert!(matches!(err, IrError::ForwardReference { at: 2, reg: 2 }));
    }

    #[test]
    fn state_index_out_of_range_is_rejected() {
        let (ops, mut a, b, imm, outputs) = ok_parts();
        a[0] = 5; // State u[5] but n_state = 1
        let err = Tape::from_parts(ops, a, b, imm, outputs, vec![], 1, 1).unwrap_err();
        assert!(matches!(
            err,
            IrError::StateIndexOutOfRange {
                at: 0,
                idx: 5,
                n_state: 1
            }
        ));
    }

    #[test]
    fn param_index_out_of_range_is_rejected() {
        let (ops, mut a, b, imm, outputs) = ok_parts();
        // Param is instruction 1; its input index lives in a[1], not b.
        a[1] = 3; // Param p[3] but n_param = 1
        let err = Tape::from_parts(ops, a, b, imm, outputs, vec![], 1, 1).unwrap_err();
        assert!(matches!(
            err,
            IrError::ParamIndexOutOfRange {
                at: 1,
                idx: 3,
                n_param: 1
            }
        ));
    }

    #[test]
    fn output_index_out_of_range_is_rejected() {
        let (ops, a, b, imm, _outputs) = ok_parts();
        let err = Tape::from_parts(ops, a, b, imm, vec![9], vec![], 1, 1).unwrap_err();
        assert!(matches!(
            err,
            IrError::OutputIndexOutOfRange {
                k: 0,
                reg: 9,
                n_reg: 3
            }
        ));
    }

    #[test]
    fn jacobian_shape_must_be_dim_squared() {
        // dim = 1 → jac must have length 1; give it 2.
        let (ops, a, b, imm, outputs) = ok_parts();
        let err = Tape::from_parts(ops, a, b, imm, outputs, vec![2, 2], 1, 1).unwrap_err();
        assert!(matches!(err, IrError::JacShapeMismatch { len: 2, dim: 1 }));
    }

    #[test]
    fn unknown_opcode_rejected_via_from_arrays() {
        let err = Tape::from_arrays(
            &[1, 2, 99],
            &[0, 0, 0],
            &[0, 0, 1],
            &[0.0; 3],
            &[2],
            &[],
            1,
            1,
        )
        .unwrap_err();
        assert_eq!(err, IrError::UnknownOpcode(99));
    }

    #[test]
    fn reachable_from_excludes_jacobian_only_registers_when_rhs_only() {
        // Tape: f(u) = u0 (output is register 0, a State leaf), plus a
        // Jacobian-only register 1 = u0 * u0 that nothing in the RHS reads.
        // outputs = [0]; jac_outputs = [1] (1x1 Jacobian = register 1).
        let ops = vec![Op::State, Op::Mul];
        let a = vec![0, 0];
        let b = vec![0, 0];
        let imm = vec![0.0, 0.0];
        let t = Tape::from_parts(ops, a, b, imm, vec![0], vec![1], 1, 0).unwrap();

        // RHS-only: only register 0 is live; the Jacobian-only Mul (reg 1) is dead.
        let rhs = t.reachable_from(false);
        assert_eq!(rhs, vec![true, false]);

        // With the Jacobian: register 1 (and the State it reads) is live too.
        let with_jac = t.reachable_from(true);
        assert_eq!(with_jac, vec![true, true]);
    }

    #[test]
    fn reachable_from_marks_full_transitive_dependency_set() {
        // f(u) = (u0 * p0) with no Jacobian: every register feeds the output.
        let (ops, a, b, imm, outputs) = ok_parts();
        let t = Tape::from_parts(ops, a, b, imm, outputs, vec![], 1, 1).unwrap();
        // State(0), Param(1), Mul(2) are all reachable from output register 2.
        assert_eq!(t.reachable_from(false), vec![true, true, true]);
        // No jac_outputs, so include_jac changes nothing.
        assert_eq!(t.reachable_from(true), vec![true, true, true]);
    }

    #[test]
    fn from_arrays_round_trips_through_ops_i32() {
        let t = Tape::from_arrays(
            &[1, 2, 12],
            &[0, 0, 0],
            &[0, 0, 1],
            &[0.0; 3],
            &[2],
            &[],
            1,
            1,
        )
        .unwrap();
        let again = Tape::from_arrays(
            &t.ops_i32(),
            t.a(),
            t.b(),
            t.imm(),
            t.outputs(),
            t.jac_outputs(),
            t.n_state(),
            t.n_param(),
        )
        .unwrap();
        assert_eq!(t, again);
    }
}
