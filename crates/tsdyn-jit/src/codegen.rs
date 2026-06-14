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
//! emitted (a backward pass), so `tsdyn_eval` never computes Jacobian-only
//! subexpressions.

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
fn math_table() -> [MathDecl; 17] {
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
    ]
}

/// Compile `tape` into a [`Compiled`] holding the finalized `eval`/`eval_jac`
/// function pointers.
pub(crate) fn compile(tape: &Tape) -> Result<Compiled, JitError> {
    let mut builder = JITBuilder::new(default_libcall_names())?;
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
    let eval_jac_id = compile_fn(&mut module, tape, true, ptr_ty, &math_ids, "tsdyn_eval_jac")?;

    module.finalize_definitions()?;

    let eval_ptr = module.get_finalized_function(eval_id);
    let eval_jac_ptr = module.get_finalized_function(eval_jac_id);
    // SAFETY: each pointer is the entry of a function we just compiled with
    // exactly the matching native signature, so the transmute to that `fn` type
    // is well-typed; `module` is returned alongside to keep the code mapped.
    let eval = unsafe { std::mem::transmute::<*const u8, EvalFn>(eval_ptr) };
    let eval_jac = unsafe { std::mem::transmute::<*const u8, EvalJacFn>(eval_jac_ptr) };

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

    let need = reachable(tape, with_jac);
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
            // Every remaining opcode is a unary transcendental (or `Sign`) lowered
            // to a host call with the same `[reg(a)]` argument.
            transcendental => {
                let x = reg(&regs, a[i]);
                call_math(module, math_ids, &mut frefs, &mut bcx, transcendental, &[x])
            }
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

/// Mark every tape register reachable from this function's outputs (and, when
/// `with_jac`, the Jacobian outputs). One backward pass: SSA references only
/// point to strictly-earlier registers, so visiting `i` after its readers
/// suffices to propagate need to its operands.
fn reachable(tape: &Tape, with_jac: bool) -> Vec<bool> {
    let n = tape.n_reg();
    let mut need = vec![false; n];

    for &r in tape.outputs() {
        need[r as usize] = true;
    }
    if with_jac {
        for &r in tape.jac_outputs() {
            need[r as usize] = true;
        }
    }

    let ops = tape.ops();
    let a = tape.a();
    let b = tape.b();
    for i in (0..n).rev() {
        if !need[i] {
            continue;
        }
        match ops[i].kind() {
            OpKind::Leaf => {}
            OpKind::Unary | OpKind::Powi => need[a[i] as usize] = true,
            OpKind::Binary => {
                need[a[i] as usize] = true;
                need[b[i] as usize] = true;
            }
        }
    }
    need
}
