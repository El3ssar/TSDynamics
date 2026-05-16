//! Right-hand side abstraction for ERK steppers.

use tsdyn_core::ir::CompiledOde;

/// `dy/dt = f(t, y)` with parameters fixed for the integration call.
pub trait Rhs {
    fn dim(&self) -> usize;
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]);
}

/// Wrapper around the bytecode IR interpreter used by N2.
pub struct IrOdeRhs<'a> {
    pub ode: &'a CompiledOde,
    pub params: &'a [f64],
    pub scratch: &'a mut Vec<f64>,
}

impl Rhs for IrOdeRhs<'_> {
    fn dim(&self) -> usize {
        self.ode.dim
    }

    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) {
        self.ode.eval_rhs(t, y, self.params, dy, self.scratch);
    }
}
