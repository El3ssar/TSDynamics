//! Prepared point/batch evaluation without integration or trajectory screening.

use tsdyn_engine::alloc::try_zeroed;
use tsdyn_engine::interrupt::Poller;
use tsdyn_ir::{Evaluator, Tape};

use super::marshal::{build_evaluator_send, EngineError};

/// Own an immutable evaluator and its code lifetime; each call owns its scratch.
pub struct PreparedEvaluator {
    evaluator: Box<dyn Evaluator + Send>,
    state_width: usize,
    input_width: usize,
    jacobian_width: usize,
}

fn allocate(rows: usize, width: usize, kind: &str) -> Result<Vec<f64>, EngineError> {
    try_zeroed(rows, width).map_err(|_| {
        EngineError::OutOfMemory(format!(
            "cannot allocate a {rows} x {width} {kind} buffer for prepared evaluation"
        ))
    })
}

impl PreparedEvaluator {
    /// Validate the argument layout and prepare the chosen evaluator once.
    pub fn new(tape: Tape, jit: bool) -> Result<Self, EngineError> {
        let input_width = tape
            .n_state()
            .checked_add(1)
            .and_then(|n| n.checked_add(tape.n_param()))
            .ok_or_else(|| EngineError::BadShape("prepared input width overflows usize".into()))?;
        if tape.dim() == 0 {
            return Err(EngineError::BadShape(
                "prepared evaluation needs at least one output".into(),
            ));
        }
        let jacobian_width = tape.jac_outputs().len();
        let state_width = tape.n_state();
        Ok(Self {
            evaluator: build_evaluator_send(tape, jit)?,
            state_width,
            input_width,
            jacobian_width,
        })
    }

    /// Width of `[state..., time, control parameters...]` for each input row.
    pub fn input_width(&self) -> usize {
        self.input_width
    }

    /// Number of primal output components.
    pub fn output_width(&self) -> usize {
        self.evaluator.dim()
    }

    /// Fused output width, refusing missing or non-square Jacobian declarations.
    pub fn result_width(&self, jacobian: bool) -> Result<usize, EngineError> {
        if !jacobian {
            return Ok(self.output_width());
        }
        if !self.evaluator.has_jacobian() {
            return Err(EngineError::BadShape(
                "prepared evaluator was compiled without a Jacobian".into(),
            ));
        }
        if self.state_width != self.evaluator.dim() {
            return Err(EngineError::BadShape(
                "prepared Jacobian evaluation requires equal state and output widths under the square tape contract".into(),
            ));
        }
        self.output_width()
            .checked_add(self.jacobian_width)
            .ok_or_else(|| EngineError::BadShape("prepared output width overflows usize".into()))
    }

    /// Evaluate contiguous logical rows, preserving every NaN/Inf component.
    ///
    /// Scratch is allocated once per call and reused only within that call.
    /// There is no live state, shared workspace, magnitude limit or row rejection.
    pub fn evaluate(&self, arguments: &[f64], jacobian: bool) -> Result<Vec<f64>, EngineError> {
        if !arguments.len().is_multiple_of(self.input_width) {
            return Err(EngineError::BadShape(format!(
                "prepared arguments must contain complete rows of width {}",
                self.input_width
            )));
        }
        let width = self.result_width(jacobian)?;
        let rows = arguments.len() / self.input_width;
        if rows == 0 {
            return Ok(Vec::new());
        }
        let mut result = allocate(rows, width, "output")?;
        let mut scratch = allocate(1, self.evaluator.n_scratch(), "scratch")?;
        let mut poll = Poller::new();
        let state_width = self.state_width;
        for (row, output) in arguments
            .chunks_exact(self.input_width)
            .zip(result.chunks_exact_mut(width))
        {
            if poll.tick() {
                return Err(EngineError::Interrupted);
            }
            let state = &row[..state_width];
            let time = row[state_width];
            let controls = &row[state_width + 1..];
            if jacobian {
                let (primal, derivative) = output.split_at_mut(self.output_width());
                self.evaluator
                    .eval_jac(state, controls, time, &mut scratch, primal, derivative);
            } else {
                self.evaluator
                    .eval(state, controls, time, &mut scratch, output);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    fn field(jacobian: bool) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let y = b.state(1);
        let a = b.param(0);
        let t = b.time();
        let ax = b.mul(a, x);
        let first = b.add(ax, t);
        let second = b.sqrt(y);
        let zero = b.constant(0.0);
        let half = b.constant(0.5);
        let slope = b.div(half, second);
        let jac = if jacobian {
            vec![a, zero, zero, slope]
        } else {
            vec![]
        };
        b.finish(&[first, second], &jac, 2, 1).unwrap()
    }

    #[test]
    fn controls_and_time_are_live_per_row() {
        for jit in [false, true] {
            let prepared = PreparedEvaluator::new(field(true), jit).unwrap();
            assert_eq!(prepared.input_width(), 4);
            assert_eq!(prepared.output_width(), 2);
            let rows = [0.5, 4.0, 0.25, 2.0, 2.0, 9.0, -1.0, 3.0];
            assert_eq!(
                prepared.evaluate(&rows, false).unwrap(),
                vec![1.25, 2.0, 5.0, 3.0]
            );
            assert_eq!(
                prepared.evaluate(&rows, true).unwrap(),
                vec![
                    1.25,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    0.25,
                    5.0,
                    3.0,
                    3.0,
                    0.0,
                    0.0,
                    1.0 / 6.0
                ]
            );
        }
    }

    #[test]
    fn nonfinite_components_do_not_erase_other_outputs() {
        for jit in [false, true] {
            let prepared = PreparedEvaluator::new(field(true), jit).unwrap();
            let result = prepared.evaluate(&[2.0, -1.0, 0.5, 3.0], true).unwrap();
            assert_eq!(result[0], 6.5);
            assert!(result[1].is_nan());
            assert_eq!(&result[2..5], &[3.0, 0.0, 0.0]);
            assert!(result[5].is_nan());
        }
    }

    #[test]
    fn missing_jacobian_and_incomplete_rows_refuse_but_empty_batches_work() {
        let prepared = PreparedEvaluator::new(field(false), false).unwrap();
        assert!(prepared.evaluate(&[], false).unwrap().is_empty());
        assert!(prepared.evaluate(&[1.0, 2.0, 3.0], false).is_err());
        assert!(prepared.evaluate(&[1.0, 2.0, 3.0, 4.0], true).is_err());
    }

    #[test]
    fn declared_state_width_sets_clock_and_parameter_offsets_for_scalar_outputs() {
        for jit in [false, true] {
            let mut b = TapeBuilder::new();
            let x = b.state(0);
            let y = b.state(1);
            let parameter = b.param(0);
            let time = b.time();
            let scaled = b.mul(parameter, x);
            let sum = b.add(scaled, y);
            let output = b.add(sum, time);
            let tape = b.finish(&[output], &[], 2, 1).unwrap();
            let prepared = PreparedEvaluator::new(tape, jit).unwrap();
            assert_eq!(
                prepared.evaluate(&[0.5, 4.0, 0.25, 2.0], false).unwrap(),
                vec![5.25]
            );
        }
    }

    #[test]
    fn non_square_declared_jacobians_refuse_before_evaluation() {
        for jit in [false, true] {
            let mut b = TapeBuilder::new();
            let y = b.state(1);
            let one = b.constant(1.0);
            let tape = b.finish(&[y], &[one], 2, 0).unwrap();
            let prepared = PreparedEvaluator::new(tape, jit).unwrap();
            let error = prepared.evaluate(&[0.5, 4.0, 0.25], true).unwrap_err();
            assert!(error.to_string().contains("equal state and output widths"));
        }
    }

    #[test]
    fn prepared_code_outlives_the_cache_entry_and_shares_without_scratch_races() {
        for jit in [false, true] {
            let prepared = PreparedEvaluator::new(field(true), jit).unwrap();
            tsdyn_jit::clear_cache();
            std::thread::scope(|scope| {
                for parameter in [1.0, 2.0, 3.0, 4.0] {
                    let evaluator = &prepared;
                    scope.spawn(move || {
                        for _ in 0..100 {
                            let result = evaluator
                                .evaluate(&[0.5, 4.0, 0.25, parameter], true)
                                .unwrap();
                            assert_eq!(result[0], 0.5 * parameter + 0.25);
                            assert_eq!(result[2], parameter);
                        }
                    });
                }
            });
        }
    }
}
