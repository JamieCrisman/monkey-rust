use std::borrow::Borrow;

use crate::code::Instructions;
use crate::evaluator::object::Object;

use super::VMError;

#[derive(Debug, Clone)]
pub struct Frame {
    // pub func: Object, // CompiledFunction
    pub instr: Instructions,
    pub num_locals: i32,
    pub ip: i64,
    pub base_pointer: i64,
}

impl Frame {
    pub fn new(instr: Instructions, num_locals: i32, base_pointer: i64) -> Self {
        Self {
            instr,
            num_locals,
            ip: -1,
            base_pointer,
        }
    }

    pub fn instructions(&self) -> Option<Instructions> {
        Some(self.instr.clone())
    }
}
