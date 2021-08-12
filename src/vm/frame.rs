use std::borrow::Borrow;

use crate::code::Instructions;
use crate::evaluator::object::Object;

use super::VMError;

#[derive(Debug, Clone)]
pub struct Frame {
    // pub func: Object, // CompiledFunction
    pub instr: Instructions,
    pub ip: i64,
}

impl Frame {
    pub fn new(func: Object) -> Result<Self, VMError> {
        match func {
            Object::CompiledFunction(instr) => Ok(Self { instr, ip: -1 }),
            _ => Err(VMError::Reason(
                "Invalid object type for frame, expected compiled function".to_string(),
            )),
        }
    }

    pub fn instructions(&self) -> Option<Instructions> {
        Some(self.instr.clone())
    }
}
