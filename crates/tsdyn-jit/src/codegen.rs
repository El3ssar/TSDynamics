//! Lowering an IR [`Tape`] to native code with Cranelift.
//!
//! [`compile`] builds one [`JITModule`] per tape holding two exported functions:
//!
//! ```text
//! tsdyn_eval     (u: *const f64, p: *const f64, t: f64, deriv: *mut f64)
//! tsdyn_eval_jac (u: *const f64, p: *const f64, t: f64, deriv: *mut f64, jac: *mut f64)
//! ```
//!
//! Both walk the tape once. Each instruction's result becomes a Cranelift SSA
//! [`Value`]; arithmetic / `Sqrt` / `Abs` lower to native IEEE-754 instructions
//! and the transcendentals / `Pow` / `Powi` / `Sign` lower to calls into the
//! host [`shims`]. Only the registers reachable from the function's outputs are
//! emitted — via the shared
//! [`Tape::reachable_from`](tsdyn_ir::Tape::reachable_from) backward pass that
//! `tsdyn-vm`'s interpreter also calls — so `tsdyn_eval` never computes
//! Jacobian-only subexpressions, and the JIT-emitted register set is *by
//! construction* the same set the interpreter executes.
//!
//! When a tape carries no Jacobian, only `tsdyn_eval` is compiled and the
//! `tsdyn_eval_jac` pointer aliases it (the bodies would be byte-identical bar an
//! unused `jac` pointer), avoiding a redundant compile.

use std::collections::HashMap;

use cranelift_codegen::ir::{types, AbiParam, FuncRef, InstBuilder, MemFlags, Value};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use tsdyn_ir::{Op, OpKind, Tape};

use crate::error::JitError;
use crate::shims;

/// The native signature of the compiled `du/dt`-only function.
pub(crate) type EvalFn = unsafe extern "C" fn(*const f64, *const f64, f64, *mut f64);
/// The native signature of the compiled `du/dt` + Jacobian function.
pub(crate) type EvalJacFn = unsafe extern "C" fn(*const f64, *const f64, f64, *mut f64, *mut f64);

/// A finalized JIT module plus the two function pointers into its code. The
/// module is kept alive (and freed on drop by the owner) so the pointers stay
/// valid.
///
/// For a tape with no Jacobian, [`compile`] does not emit a separate
/// `tsdyn_eval_jac` body — `eval_jac` then aliases the `tsdyn_eval` entry (the
/// extra `jac` pointer is never read), so the field is always a callable pointer
/// regardless of whether the tape carries a Jacobian.
pub(crate) struct Compiled {
    pub(crate) module: JITModule,
    pub(crate) eval: EvalFn,
    pub(crate) eval_jac: EvalJacFn,
}

/// The ABI of a host math shim.
#[derive(Clone, Copy)]
enum Arity {
    /// `(f64) -> f64`.
    Unary,
    /// `(f64, f64) -> f64`.
    Binary,
    /// `(f64, i32) -> f64`.
    Powi,
}

/// One opcode lowered to a host call: which [`Op`], the symbol name to register
/// it under, the shim's address, and its ABI.
struct MathDecl {
    op: Op,
    name: &'static str,
    addr: *const u8,
    arity: Arity,
}

/// Every opcode that lowers to a host call, paired with its shim. Opcodes not
/// listed here (the leaves, arithmetic, `Sqrt`, `Abs`) lower to native Cranelift
/// instructions.
fn math_table() -> [MathDecl; 29] {
    macro_rules! d {
        ($op:expr, $name:literal, $shim:path, $arity:expr) => {
            MathDecl {
                op: $op,
                name: $name,
                addr: $shim as *const u8,
                arity: $arity,
            }
        };
    }
    [
        d!(Op::Pow, "tsdyn_pow", shims::pow, Arity::Binary),
        d!(Op::Powi, "tsdyn_powi", shims::powi, Arity::Powi),
        d!(Op::Sign, "tsdyn_sign", shims::sign, Arity::Unary),
        d!(Op::Sin, "tsdyn_sin", shims::sin, Arity::Unary),
        d!(Op::Cos, "tsdyn_cos", shims::cos, Arity::Unary),
        d!(Op::Tan, "tsdyn_tan", shims::tan, Arity::Unary),
        d!(Op::Exp, "tsdyn_exp", shims::exp, Arity::Unary),
        d!(Op::Log, "tsdyn_log", shims::log, Arity::Unary),
        d!(Op::Sinh, "tsdyn_sinh", shims::sinh, Arity::Unary),
        d!(Op::Cosh, "tsdyn_cosh", shims::cosh, Arity::Unary),
        d!(Op::Tanh, "tsdyn_tanh", shims::tanh, Arity::Unary),
        d!(Op::Asin, "tsdyn_asin", shims::asin, Arity::Unary),
        d!(Op::Acos, "tsdyn_acos", shims::acos, Arity::Unary),
        d!(Op::Atan, "tsdyn_atan", shims::atan, Arity::Unary),
        d!(Op::Asinh, "tsdyn_asinh", shims::asinh, Arity::Unary),
        d!(Op::Acosh, "tsdyn_acosh", shims::acosh, Arity::Unary),
        d!(Op::Atanh, "tsdyn_atanh", shims::atanh, Arity::Unary),
        // Non-smooth / piecewise block (stream E-OPS).
        d!(Op::Lt, "tsdyn_lt", shims::lt, Arity::Binary),
        d!(Op::Le, "tsdyn_le", shims::le, Arity::Binary),
        d!(Op::Gt, "tsdyn_gt", shims::gt, Arity::Binary),
        d!(Op::Ge, "tsdyn_ge", shims::ge, Arity::Binary),
        d!(Op::Eq, "tsdyn_eq", shims::eq, Arity::Binary),
        d!(Op::Ne, "tsdyn_ne", shims::ne, Arity::Binary),
        d!(Op::Min, "tsdyn_min", shims::min, Arity::Binary),
        d!(Op::Max, "tsdyn_max", shims::max, Arity::Binary),
        d!(Op::Floor, "tsdyn_floor", shims::floor, Arity::Unary),
        d!(Op::Ceil, "tsdyn_ceil", shims::ceil, Arity::Unary),
        d!(Op::Mod, "tsdyn_mod", shims::modulo, Arity::Binary),
        d!(Op::Rem, "tsdyn_rem", shims::rem, Arity::Binary),
    ]
}

/// Build the [`JITBuilder`] every tape is compiled through, asking Cranelift for
/// `opt_level = "speed"` rather than its default `"none"`.
///
/// [`JITBuilder::new`] leaves Cranelift at `opt_level = "none"` (minimise
/// *compile* time by disabling most optimisations). `"speed"` ("generate the
/// fastest possible code") is the I-BENCH report's C4 recommendation and the
/// intent-correct default for a native-code evaluator — but see the measured
/// effect below before reading it as a speed-up.
///
/// **Measured effect: neutral on the benched workloads.** An interleaved A/B
/// (`none` vs `speed`, drift-cancelled) on the compute-bound 128-dim Lorenz-96
/// ODE and the Hénon map showed *no* runtime difference beyond noise — `jit`
/// stays a tie with gcc-`-O3` JiTCODE either way. The tapes lower to already-CSE'd
/// straight-line SSA with no loops or redundancy for Cranelift's GVN/LICM/
/// instruction-combining to exploit, and the solver's per-step vector arithmetic
/// (the other half of a large ODE step) is rustc-compiled, outside Cranelift's
/// reach. The flag is kept because it is the correct intent, costs only a sub-ms
/// of extra one-time compile (verified by `compile_latency`), is bit-for-bit safe
/// (below), and *may* help arithmetic-heavy tapes outside the bench sample; it is
/// not claimed as a speed-up on the measured cases.
///
/// We route through [`JITBuilder::with_flags`] rather than hand-building an ISA so
/// the JIT's two non-negotiable relocation flags stay correct: `with_flags`
/// applies our `opt_level` *first*, then force-sets `use_colocated_libcalls=false`
/// and `is_pic=false` (which [`cranelift_jit::JITModule`] asserts) before building
/// the host ISA via the same `cranelift-native` detection [`JITBuilder::new`] uses.
/// So this changes *only* the optimisation level — no new dependency, still pure
/// Rust / no LLVM, and the host-CPU feature set is unchanged.
///
/// # Numerical contract preserved
///
/// `opt_level = "speed"` enables Cranelift's general optimisation passes (GVN,
/// LICM, instruction combining, register allocation); it does **not** enable
/// floating-point contraction (FMA), reassociation or any fast-math relaxation,
/// so the natively-lowered IEEE-754 ops stay bit-identical to the interpreter's
/// Rust operators. The `interpreter_equivalence` fuzz test re-asserts this
/// bit-for-bit equality at the new opt level.
fn build_jit_builder() -> Result<JITBuilder, JitError> {
    Ok(JITBuilder::with_flags(
        &[("opt_level", "speed")],
        default_libcall_names(),
    )?)
}

/// Compile `tape` into a [`Compiled`] holding the finalized `eval`/`eval_jac`
/// function pointers.
pub(crate) fn compile(tape: &Tape) -> Result<Compiled, JitError> {
    let mut builder = build_jit_builder()?;
    let table = math_table();
    // Register every shim's address with the JIT before the module is created
    // (symbols cannot be added afterwards).
    for decl in &table {
        builder.symbol(decl.name, decl.addr);
    }

    let mut module = JITModule::new(builder);

    // Declare every shim as an imported function, recording its FuncId. Declaring
    // an unused import is harmless — it is only relocated if actually called.
    let mut math_ids: HashMap<Op, FuncId> = HashMap::new();
    for decl in &table {
        let mut sig = module.make_signature();
        match decl.arity {
            Arity::Unary => sig.params.push(AbiParam::new(types::F64)),
            Arity::Binary => {
                sig.params.push(AbiParam::new(types::F64));
                sig.params.push(AbiParam::new(types::F64));
            }
            Arity::Powi => {
                sig.params.push(AbiParam::new(types::F64));
                sig.params.push(AbiParam::new(types::I32));
            }
        }
        sig.returns.push(AbiParam::new(types::F64));
        let id = module.declare_function(decl.name, Linkage::Import, &sig)?;
        math_ids.insert(decl.op, id);
    }

    let ptr_ty = module.target_config().pointer_type();
    let eval_id = compile_fn(&mut module, tape, false, ptr_ty, &math_ids, "tsdyn_eval")?;
    // Only compile a distinct `tsdyn_eval_jac` body when the tape actually carries
    // a Jacobian. For a Jacobian-free tape the two bodies would be byte-identical
    // except for an extra, never-written `jac` pointer, so we skip the redundant
    // compile and alias `eval_jac` to the `tsdyn_eval` entry below — saving a
    // sub-ms of one-time compile (bounded by `compile_latency`). The contract is
    // preserved: a no-Jacobian tape's `eval_jac` leaves `jac` untouched either
    // way (the aliased `eval` never reads or writes the extra pointer).
    let eval_jac_id = if tape.has_jacobian() {
        Some(compile_fn(
            &mut module,
            tape,
            true,
            ptr_ty,
            &math_ids,
            "tsdyn_eval_jac",
        )?)
    } else {
        None
    };

    module.finalize_definitions()?;

    let eval_ptr = module.get_finalized_function(eval_id);
    // SAFETY: `eval_ptr` is the entry of the function we just compiled with the
    // matching native signature, so the transmute to `EvalFn` is well-typed;
    // `module` is returned alongside to keep the code mapped.
    let eval = unsafe { std::mem::transmute::<*const u8, EvalFn>(eval_ptr) };
    let eval_jac = match eval_jac_id {
        // SAFETY: a real Jacobian body compiled with the `EvalJacFn` signature.
        Some(id) => {
            let p = module.get_finalized_function(id);
            unsafe { std::mem::transmute::<*const u8, EvalJacFn>(p) }
        }
        // SAFETY: no Jacobian to compute — alias the `tsdyn_eval` entry. The
        // `EvalJacFn` ABI is `EvalFn` plus one trailing `*mut f64` (the `jac`
        // pointer); under the C calling convention an extra ignored pointer
        // argument is benign, and `JitEvaluator::eval_jac` writes nothing to
        // `jac` for a Jacobian-free tape, so the callee never reads it.
        None => unsafe { std::mem::transmute::<*const u8, EvalJacFn>(eval_ptr) },
    };

    Ok(Compiled {
        module,
        eval,
        eval_jac,
    })
}

/// Declare and define one tape function, returning its [`FuncId`].
fn compile_fn(
    module: &mut JITModule,
    tape: &Tape,
    with_jac: bool,
    ptr_ty: types::Type,
    math_ids: &HashMap<Op, FuncId>,
    name: &str,
) -> Result<FuncId, JitError> {
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_ty)); // u
    sig.params.push(AbiParam::new(ptr_ty)); // p
    sig.params.push(AbiParam::new(types::F64)); // t
    sig.params.push(AbiParam::new(ptr_ty)); // deriv
    if with_jac {
        sig.params.push(AbiParam::new(ptr_ty)); // jac
    }

    let func_id = module.declare_function(name, Linkage::Export, &sig)?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let mut fctx = FunctionBuilderContext::new();
    build_body(module, tape, with_jac, math_ids, &mut ctx, &mut fctx);

    module.define_function(func_id, &mut ctx)?;
    module.clear_context(&mut ctx);
    Ok(func_id)
}

/// Emit the Cranelift IR body of one tape function into `ctx.func`.
fn build_body(
    module: &mut JITModule,
    tape: &Tape,
    with_jac: bool,
    math_ids: &HashMap<Op, FuncId>,
    ctx: &mut Context,
    fctx: &mut FunctionBuilderContext,
) {
    let mut bcx = FunctionBuilder::new(&mut ctx.func, fctx);

    let entry = bcx.create_block();
    bcx.append_block_params_for_function_params(entry);
    bcx.switch_to_block(entry);
    bcx.seal_block(entry);

    let params = bcx.block_params(entry).to_vec();
    let u_ptr = params[0];
    let p_ptr = params[1];
    let t_val = params[2];
    let deriv_ptr = params[3];
    let jac_ptr = if with_jac { Some(params[4]) } else { None };

    // Shared dead-register-elimination pass (hoisted into `tsdyn-ir`): emit only
    // the registers this function's outputs reach. `tsdyn_eval` (`with_jac =
    // false`) drops Jacobian-only subexpressions; `tsdyn_eval_jac` includes them.
    // The interpreter (`tsdyn-vm`) calls the *same* `reachable_from(false)`, so
    // the JIT-emitted and interpreter-executed register sets are structurally one
    // algorithm, never two parallel copies.
    let need = tape.reachable_from(with_jac);
    let n = tape.n_reg();
    // Each tape register maps to the SSA Value that computes it. `None` means the
    // register was not reachable from this function's outputs (so never emitted);
    // reachability guarantees any register read by a needed instruction is `Some`.
    let mut regs: Vec<Option<Value>> = vec![None; n];
    let mut frefs: HashMap<Op, FuncRef> = HashMap::new();
    let flags = MemFlags::trusted();

    let ops = tape.ops();
    let a = tape.a();
    let b = tape.b();
    let imm = tape.imm();

    for i in 0..n {
        if !need[i] {
            continue;
        }
        let ai = a[i] as usize;
        // INVARIANT: the natively-lowered arithmetic ops below must stay plain
        // IEEE-754 instructions — no FMA contraction, no reassociation, no
        // fast-math flags — so they are bit-identical to the Rust operators the
        // interpreter uses. Cranelift does this by default (it never contracts
        // unless an `fma` is emitted explicitly); do not enable a contract /
        // fast-math flag here without breaking the bit-for-bit equality the
        // `interpreter_equivalence` test guards.
        let value = match ops[i] {
            Op::Const => bcx.ins().f64const(imm[i]),
            Op::State => bcx.ins().load(types::F64, flags, u_ptr, offset(ai)),
            Op::Param => bcx.ins().load(types::F64, flags, p_ptr, offset(ai)),
            Op::Time => t_val,
            Op::Add => {
                let (x, y) = (reg(&regs, a[i]), reg(&regs, b[i]));
                bcx.ins().fadd(x, y)
            }
            Op::Sub => {
                let (x, y) = (reg(&regs, a[i]), reg(&regs, b[i]));
                bcx.ins().fsub(x, y)
            }
            Op::Mul => {
                let (x, y) = (reg(&regs, a[i]), reg(&regs, b[i]));
                bcx.ins().fmul(x, y)
            }
            Op::Div => {
                let (x, y) = (reg(&regs, a[i]), reg(&regs, b[i]));
                bcx.ins().fdiv(x, y)
            }
            Op::Neg => {
                let x = reg(&regs, a[i]);
                bcx.ins().fneg(x)
            }
            Op::Recip => {
                let one = bcx.ins().f64const(1.0);
                let x = reg(&regs, a[i]);
                bcx.ins().fdiv(one, x)
            }
            Op::Sqrt => {
                let x = reg(&regs, a[i]);
                bcx.ins().sqrt(x)
            }
            Op::Abs => {
                let x = reg(&regs, a[i]);
                bcx.ins().fabs(x)
            }
            Op::Pow => {
                let (x, y) = (reg(&regs, a[i]), reg(&regs, b[i]));
                call_math(module, math_ids, &mut frefs, &mut bcx, Op::Pow, &[x, y])
            }
            Op::Powi => {
                let x = reg(&regs, a[i]);
                let e = bcx.ins().iconst(types::I32, b[i] as i64);
                call_math(module, math_ids, &mut frefs, &mut bcx, Op::Powi, &[x, e])
            }
            // Every remaining opcode lowers to a host shim: the unary
            // transcendentals / `Sign` / `Floor` / `Ceil` take `[reg(a)]`, the
            // comparisons / `Min` / `Max` / `Mod` / `Rem` take `[reg(a), reg(b)]`.
            // Dispatching by arity keeps them bit-identical to the interpreter.
            shim => match shim.kind() {
                OpKind::Unary => {
                    let x = reg(&regs, a[i]);
                    call_math(module, math_ids, &mut frefs, &mut bcx, shim, &[x])
                }
                OpKind::Binary => {
                    let (x, y) = (reg(&regs, a[i]), reg(&regs, b[i]));
                    call_math(module, math_ids, &mut frefs, &mut bcx, shim, &[x, y])
                }
                // Leaves and `Powi` are handled by explicit arms above.
                OpKind::Leaf | OpKind::Powi => {
                    unreachable!("leaf/powi handled by explicit arms: {shim:?}")
                }
            },
        };
        regs[i] = Some(value);
    }

    for (k, &slot) in tape.outputs().iter().enumerate() {
        let v = reg(&regs, slot);
        bcx.ins().store(flags, v, deriv_ptr, offset(k));
    }
    if let Some(jp) = jac_ptr {
        for (m, &slot) in tape.jac_outputs().iter().enumerate() {
            let v = reg(&regs, slot);
            bcx.ins().store(flags, v, jp, offset(m));
        }
    }

    bcx.ins().return_(&[]);
    bcx.finalize();
}

/// Byte offset of the `index`-th `f64` in a packed array.
#[inline]
fn offset(index: usize) -> i32 {
    (index * std::mem::size_of::<f64>()) as i32
}

/// The SSA value computing tape register `idx`. Panics only on a reachability
/// bug (a needed instruction read a register that was never emitted).
#[inline]
fn reg(regs: &[Option<Value>], idx: i32) -> Value {
    regs[idx as usize].expect("operand register not emitted (reachability invariant violated)")
}

/// Emit a call to the host shim for `op`, importing it into the current function
/// on first use, and return the result value.
fn call_math(
    module: &mut JITModule,
    math_ids: &HashMap<Op, FuncId>,
    cache: &mut HashMap<Op, FuncRef>,
    bcx: &mut FunctionBuilder,
    op: Op,
    args: &[Value],
) -> Value {
    let fref = match cache.get(&op) {
        Some(&fr) => fr,
        None => {
            let id = math_ids[&op];
            let fr = module.declare_func_in_func(id, bcx.func);
            cache.insert(op, fr);
            fr
        }
    };
    let call = bcx.ins().call(fref, args);
    bcx.inst_results(call)[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    /// f(u) = u0 * u0, no Jacobian — exercises the `eval_jac`-aliasing path.
    fn square_no_jac() -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let sq = b.mul(x, x);
        b.finish(&[sq], &[], 1, 0).unwrap()
    }

    #[test]
    fn no_jacobian_tape_aliases_eval_jac_to_eval() {
        // A Jacobian-free tape compiles only `tsdyn_eval`; `eval_jac` then aliases
        // it. Calling the aliased `eval_jac` (5-arg signature) must compute the
        // same `deriv` as `eval` and never touch the trailing `jac` pointer.
        let compiled = compile(&square_no_jac()).unwrap();
        let u = [3.0_f64];
        let p: [f64; 0] = [];

        let mut deriv_eval = [0.0_f64];
        // SAFETY: matching 4-arg signature; buffers are correctly sized.
        unsafe {
            (compiled.eval)(u.as_ptr(), p.as_ptr(), 0.0, deriv_eval.as_mut_ptr());
        }
        assert_eq!(deriv_eval[0], 9.0);

        let mut deriv_jac = [0.0_f64];
        let mut jac = [f64::NAN]; // sentinel: must stay untouched
                                  // SAFETY: `eval_jac` aliases the 4-arg `tsdyn_eval`; the extra `jac`
                                  // pointer is passed per the C ABI and never read by the callee.
        unsafe {
            (compiled.eval_jac)(
                u.as_ptr(),
                p.as_ptr(),
                0.0,
                deriv_jac.as_mut_ptr(),
                jac.as_mut_ptr(),
            );
        }
        assert_eq!(deriv_jac[0], 9.0, "aliased eval_jac must match eval");
        assert!(jac[0].is_nan(), "no-Jacobian tape must leave jac untouched");
    }

    #[test]
    fn jacobian_tape_compiles_a_distinct_eval_jac() {
        // With a Jacobian, a real `tsdyn_eval_jac` is compiled and writes the
        // Jacobian entries. f(u) = u0^2 → f'(u) = 2*u0.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let sq = b.mul(x, x);
        let two = b.constant(2.0);
        let dfdx = b.mul(two, x); // ∂f/∂u0 = 2*u0
        let tape = b.finish(&[sq], &[dfdx], 1, 0).unwrap();
        assert!(tape.has_jacobian());

        let compiled = compile(&tape).unwrap();
        let u = [3.0_f64];
        let p: [f64; 0] = [];
        let mut deriv = [0.0_f64];
        let mut jac = [0.0_f64];
        // SAFETY: matching 5-arg signature; buffers sized dim and dim*dim.
        unsafe {
            (compiled.eval_jac)(
                u.as_ptr(),
                p.as_ptr(),
                0.0,
                deriv.as_mut_ptr(),
                jac.as_mut_ptr(),
            );
        }
        assert_eq!(deriv[0], 9.0);
        assert_eq!(jac[0], 6.0, "Jacobian 2*u0 at u0=3");
    }

    #[test]
    fn jit_reachability_matches_shared_tape_pass() {
        // The codegen reachability is now `Tape::reachable_from`; pin that the
        // RHS-only seed (with_jac = false) drops the Jacobian-only register.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let sq = b.mul(x, x); // RHS output (reg used by output)
        let two = b.constant(2.0);
        let dfdx = b.mul(two, x); // Jacobian-only
        let tape = b.finish(&[sq], &[dfdx], 1, 0).unwrap();

        let rhs_only = tape.reachable_from(false);
        let with_jac = tape.reachable_from(true);
        // The Jacobian register and the constant `2.0` it reads are live only
        // with the Jacobian seeded.
        assert!(
            with_jac.iter().filter(|&&b| b).count() > rhs_only.iter().filter(|&&b| b).count(),
            "the Jacobian path must keep strictly more registers live"
        );
    }
}
