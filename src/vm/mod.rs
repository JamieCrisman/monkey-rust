mod frame;
use std::{borrow::Borrow, collections::HashMap, ops::Deref};

use crate::{
    code::{Instructions, Opcode},
    compiler::Bytecode,
    evaluator::object::{Object, ObjectType},
};

use self::frame::Frame;

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum VMError {
    Reason(String),
}

const DEFAULT_STACK_SIZE: usize = 2048;
pub const GLOBALS_SIZE: usize = 65536;

const TRUE: Object = Object::Bool(true);
const FALSE: Object = Object::Bool(false);
const NULL: Object = Object::Null;

fn is_truthy(obj: Object) -> bool {
    match obj {
        Object::Bool(b) => b,
        Object::Null => false, // ?
        _ => true,
    }
}

pub struct VM<'a> {
    constants: Vec<Object>,
    // instructions: Instructions,
    stack: Vec<Object>,
    sp: usize,
    pub globals: &'a mut Vec<Object>,
    // stack_size: i32,
    frames: Vec<Frame>,
    frames_index: i32,
}

impl<'a> VM<'a> {
    pub fn new_with_global_store(bytecode: Bytecode, g: &'a mut Vec<Object>) -> Self {
        let mut frames = vec![
            Frame::new(Object::CompiledFunction(bytecode.instructions.clone()))
                .expect("expected a valid main"),
        ];
        frames[0].ip += 1;
        Self {
            // instructions: bytecode.instructions.clone(),
            constants: bytecode.constants.clone(),
            sp: 0,
            stack: Vec::with_capacity(DEFAULT_STACK_SIZE),
            globals: g,
            frames,
            frames_index: 1,
        }
    }

    fn current_frame(&mut self) -> &Frame {
        return self
            .frames
            .get((self.frames_index - 1) as usize)
            .as_ref()
            .unwrap();
    }

    fn set_ip(&mut self, new_ip: i64) {
        self.frames[(self.frames_index - 1) as usize].ip = new_ip;
    }

    fn push_frame(&mut self, f: Frame) {
        self.frames.push(f);
        self.frames_index += 1;
    }

    fn pop_frame(&mut self) -> Frame {
        self.frames_index -= 1;
        return self.frames.pop().unwrap();
    }

    pub fn stack_top(&self) -> Option<Object> {
        if self.sp == 0 {
            return None;
        }

        let ob = self
            .stack
            .get(self.sp - 1)
            .expect("expected result")
            .clone();
        return Some(ob);
    }

    pub fn run(&mut self) -> Result<(), VMError> {
        // for instr in self.instructions.data.iter() {

        // }
        // let mut ip: usize = self.current_frame().ip as usize;
        while (self.current_frame().ip as usize)
            < self
                .current_frame()
                .instructions()
                .expect("expected instructions")
                .data
                .len()
        {
            // let a = self.current_frame().ip;
            // println!(
            //     "ip {} of {}",
            //     a,
            //     self.current_frame()
            //         .instructions()
            //         .expect("expected instructions")
            //         .data
            //         .len()
            // );
            let ip = self.current_frame().ip as usize;
            // self.set_ip(ip as i64);
            let op = Opcode::from(
                *self
                    .current_frame()
                    .instructions()
                    .expect("expected instructions")
                    .data
                    .get(ip)
                    .expect("expected byte"),
            );
            // println!(" ------- got opcode: {:?} ip: {}", op, ip);
            let mut cur_instructions = self
                .current_frame()
                .instructions()
                .expect("expected instructions");
            // println!("{:?}", cur_instructions.data);
            match op {
                Opcode::Constant => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    // println!("ip 1,2 {},{}", ip + 1, ip + 2);
                    let const_index = u16::from_be_bytes(buff);
                    let new_ip = self.current_frame().ip + op.operand_width() as i64;
                    self.set_ip(new_ip);
                    // println!("got const from index: {}, new ip {}", const_index, new_ip);
                    // println!("actual curr ip {}", self.current_frame().ip);
                    // println!("instructions {:?}", cur_instructions.data);
                    // println!("constants {:?}", self.constants);
                    self.push(self.constants.get(const_index as usize).unwrap().clone())?;
                }
                Opcode::Add | Opcode::Divide | Opcode::Multiply | Opcode::Subtract => {
                    self.execute_binary_operation(op)?;
                }
                Opcode::True => self.push(TRUE)?,
                Opcode::False => self.push(FALSE)?,
                Opcode::Equal | Opcode::NotEqual | Opcode::GreaterThan => {
                    self.execute_comparison(op)?;
                }
                Opcode::Bang => {
                    self.execute_bang_operator()?;
                }
                Opcode::Minus => {
                    self.execute_minus_operator()?;
                }
                Opcode::Jump => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let jump_target = u16::from_be_bytes(buff);
                    self.set_ip(jump_target as i64 - 1);
                }
                Opcode::JumpNotTruthy => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let jump_target = u16::from_be_bytes(buff);
                    self.set_ip((ip + 2) as i64);
                    let condition = self.pop();
                    if !is_truthy(condition) {
                        self.set_ip((jump_target - 1) as i64);
                    }
                }
                Opcode::Pop => {
                    self.pop();
                }
                Opcode::Null => self.push(NULL)?,
                Opcode::GetGlobal => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let global_index = u16::from_be_bytes(buff);
                    self.set_ip((ip + 2) as i64);
                    let val = (*self.globals.get(global_index as usize).unwrap()).clone();
                    self.push(val)?;
                }
                Opcode::SetGlobal => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let global_index = u16::from_be_bytes(buff);
                    self.set_ip((ip + 2) as i64);
                    let pop = self.pop();
                    self.globals.insert(global_index as usize, pop);
                }
                Opcode::Array => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let element_count = u16::from_be_bytes(buff);
                    self.set_ip((ip + 2) as i64);
                    let array = self.build_array(self.sp - element_count as usize, self.sp);
                    self.push(array)?;
                }
                Opcode::Hash => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let element_count = u16::from_be_bytes(buff);
                    self.set_ip((ip + 2) as i64);
                    let hash = self.build_hash(self.sp - element_count as usize, self.sp);
                    self.sp -= element_count as usize;
                    self.push(hash)?;
                }
                Opcode::Index => {
                    let index = self.pop();
                    let left = self.pop();
                    self.execute_index_expression(left, index)?;
                }
                Opcode::Call => {
                    let func = match self.stack[self.sp - 1].clone() {
                        Object::CompiledFunction(instr) => instr,
                        _ => return Err(VMError::Reason("expected function".to_string())),
                    };
                    self.push_frame(
                        Frame::new(Object::CompiledFunction(func))
                            .expect("expected to create new frame"),
                    );
                    // cur_instructions = self
                    //     .current_frame()
                    //     .instructions()
                    //     .expect("expected instructions");
                }
                Opcode::Return => {
                    self.pop_frame();
                    self.pop();
                    self.push(Object::Null)?
                }
                Opcode::ReturnValue => {
                    let ret_val = self.pop();
                    self.pop_frame();
                    self.pop();
                    self.push(ret_val)?;
                    // cur_instructions = self
                    //     .current_frame()
                    //     .instructions()
                    //     .expect("expected instructions");
                }
                Opcode::GetLocal => {}
                Opcode::SetLocal => {}
            }

            let next_ip = self.current_frame().ip + 1;
            self.set_ip(next_ip);
            // println!("ending at ---- {}", self.current_frame().ip);
        }

        Ok(())
    }

    fn execute_index_expression(&mut self, left: Object, index: Object) -> Result<(), VMError> {
        if left.object_type() == ObjectType::Array && index.object_type() == ObjectType::Int {
            return self.execute_array_index(left, index);
        } else if left.object_type() == ObjectType::Hash {
            return self.execute_hash_index(left, index);
        } else {
            return Err(VMError::Reason(format!(
                "index operator not supported for {:?}",
                left.object_type()
            )));
        }
    }

    fn execute_array_index(&mut self, left: Object, index: Object) -> Result<(), VMError> {
        let index_val = match index {
            Object::Int(i) => i,
            _ => {
                return Err(VMError::Reason(format!(
                    "expected array type, but got {:?}",
                    left.object_type()
                )))
            }
        };

        match left {
            Object::Array(a) => {
                let max = (a.len() - 1) as i64;
                if index_val < 0 || index_val > max {
                    return self.push(Object::Null);
                }

                return self.push(
                    a.get(index_val as usize)
                        .expect("expected a value from index")
                        .clone(),
                );
            }
            _ => {
                return Err(VMError::Reason(format!(
                    "expected array type, but got {:?}",
                    left.object_type()
                )))
            }
        }
    }

    fn execute_hash_index(&mut self, left: Object, index: Object) -> Result<(), VMError> {
        match left {
            Object::Hash(h) => {
                let val = match h.get(&index) {
                    Some(v) => v.clone(),
                    None => Object::Null,
                };
                return self.push(val);
            }
            _ => {
                return Err(VMError::Reason(format!(
                    "expected array type, but got {:?}",
                    left.object_type()
                )))
            }
        }
    }

    fn build_hash(&mut self, start_index: usize, end_index: usize) -> Object {
        let mut map: HashMap<Object, Object> = HashMap::new();

        let mut i = start_index;
        while i < end_index {
            let key: Object = self
                .stack
                .get(i)
                .expect("expected a value from stack")
                .clone();
            let value: Object = self
                .stack
                .get(i + 1)
                .expect("expected a value from stack")
                .clone();
            map.insert(key, value);

            i += 2;
        }

        Object::Hash(map)
    }

    fn build_array(&mut self, start_index: usize, end_index: usize) -> Object {
        let mut elements: Vec<Object> = vec![];
        if start_index != end_index {
            for pos in start_index..end_index {
                let item = self
                    .stack
                    .get(pos)
                    .expect("expected a valid index position")
                    .clone();
                elements.push(item);
            }
        }

        return Object::Array(elements);
    }

    fn execute_bang_operator(&mut self) -> Result<(), VMError> {
        let op = self.pop();

        match op {
            Object::Bool(true) => self.push(FALSE),
            Object::Bool(false) => self.push(TRUE),
            Object::Null => self.push(TRUE),
            _ => self.push(FALSE),
        }
    }

    fn execute_minus_operator(&mut self) -> Result<(), VMError> {
        let op = self.pop();

        match op {
            Object::Int(int) => self.push(Object::Int(-int)),
            _ => Err(VMError::Reason(format!(
                "unsupported minus type: {:?} for {:?}",
                op.object_type(),
                op,
            ))),
        }
    }

    fn execute_comparison(&mut self, op: Opcode) -> Result<(), VMError> {
        let right = self.pop();
        let left = self.pop();

        match (left.object_type(), right.object_type()) {
            (ObjectType::Int, ObjectType::Int) => {
                return self.execute_integer_comparison(op, left, right);
            }
            (ObjectType::Bool, ObjectType::Bool) => {}
            (a, b) => {
                return Err(VMError::Reason(format!(
                    "Unsupported binary action for {:?} and {:?}",
                    a, b
                )))
            }
        };

        let right_bool = match right {
            Object::Bool(b) => b,
            _ => return Err(VMError::Reason("Unexpected comparison type".to_string())),
        };
        let left_bool = match left {
            Object::Bool(b) => b,
            _ => return Err(VMError::Reason("Unexpected comparison type".to_string())),
        };

        match op {
            Opcode::Equal => match left_bool == right_bool {
                true => return self.push(TRUE),
                false => return self.push(FALSE),
            },
            Opcode::NotEqual => match left_bool != right_bool {
                true => return self.push(TRUE),
                false => return self.push(FALSE),
            },
            _ => {
                return Err(VMError::Reason(format!(
                    "Unsupported operator action ({:?}) for {:?} and {:?}",
                    op, left, right
                )))
            }
        }

        // Ok(())
    }

    fn execute_integer_comparison(
        &mut self,
        op: Opcode,
        left: Object,
        right: Object,
    ) -> Result<(), VMError> {
        let left_val = match left {
            Object::Int(i) => i,
            _ => return Err(VMError::Reason("Unexpected comparison type".to_string())),
        };
        let right_val = match right {
            Object::Int(i) => i,
            _ => return Err(VMError::Reason("Unexpected comparison type".to_string())),
        };

        match op {
            Opcode::Equal => match left_val == right_val {
                true => return self.push(TRUE),
                false => return self.push(FALSE),
            },
            Opcode::NotEqual => match left_val != right_val {
                true => return self.push(TRUE),
                false => return self.push(FALSE),
            },
            Opcode::GreaterThan => match left_val > right_val {
                true => return self.push(TRUE),
                false => return self.push(FALSE),
            },
            _ => {
                return Err(VMError::Reason(format!(
                    "Unsupported operator action ({:?}) for {:?} and {:?}",
                    op, left, right
                )))
            }
        }
    }

    fn execute_binary_operation(&mut self, op: Opcode) -> Result<(), VMError> {
        let right = self.pop();
        let left = self.pop();

        match (left.object_type(), right.object_type()) {
            (ObjectType::Int, ObjectType::Int) => {
                self.execute_binary_integer_operation(op, left, right)?;
            }
            (ObjectType::String, ObjectType::String) => {
                self.execute_binary_string_operation(op, left, right)?;
            }
            (a, b) => {
                return Err(VMError::Reason(format!(
                    "Unsupported binary action for {:?} and {:?}",
                    a, b
                )))
            }
        };

        // self.push(Object::Int(left + right))?;

        Ok(())
    }

    fn execute_binary_string_operation(
        &mut self,
        op: Opcode,
        left: Object,
        right: Object,
    ) -> Result<(), VMError> {
        let left_val = match left {
            Object::String(s) => s,
            _ => return Err(VMError::Reason("Unexpected type".to_string())),
        };
        let right_val = match right {
            Object::String(s) => s,
            _ => return Err(VMError::Reason("Unexpected type".to_string())),
        };

        match op {
            Opcode::Add => {
                self.push(Object::String(format!("{}{}", left_val, right_val)))?;
            }
            // Opcode::Divide => {
            //     self.push(Object::Int(left_val / right_val))?;
            // }
            // Opcode::Multiply => {
            //     self.push(Object::Int(left_val * right_val))?;
            // }
            // Opcode::Subtract => {
            //     self.push(Object::Int(left_val - right_val))?;
            // }
            _ => return Err(VMError::Reason("Unexpected string operation".to_string())),
        }

        Ok(())
    }

    fn execute_binary_integer_operation(
        &mut self,
        op: Opcode,
        left: Object,
        right: Object,
    ) -> Result<(), VMError> {
        let left_val = match left {
            Object::Int(int) => int,
            _ => return Err(VMError::Reason("Unexpected type".to_string())),
        };
        let right_val = match right {
            Object::Int(int) => int,
            _ => return Err(VMError::Reason("Unexpected type".to_string())),
        };

        match op {
            Opcode::Add => {
                self.push(Object::Int(left_val + right_val))?;
            }
            Opcode::Divide => {
                self.push(Object::Int(left_val / right_val))?;
            }
            Opcode::Multiply => {
                self.push(Object::Int(left_val * right_val))?;
            }
            Opcode::Subtract => {
                self.push(Object::Int(left_val - right_val))?;
            }
            _ => return Err(VMError::Reason("Unexpected operation".to_string())),
        }

        Ok(())
    }

    fn push(&mut self, obj: Object) -> Result<(), VMError> {
        if self.sp >= self.stack.capacity() {
            return Err(VMError::Reason("Stack overflow".to_string()));
        }

        self.stack.insert(self.sp, obj);
        self.sp += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Object {
        self.sp -= 1;
        self.stack
            .get(self.sp)
            .expect("Expected something to be on the stack")
            .clone()
    }

    pub fn last_popped(&self) -> Option<Object> {
        match self.stack.get(self.sp) {
            Some(obj) => Some(obj.clone()),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    // use crate::code::*;
    use crate::compiler::symbol_table::SymbolTable;
    use crate::compiler::*;
    use crate::evaluator::object::*;
    use crate::lexer;
    use crate::parser;
    use crate::vm::VM;
    use std::collections::HashMap;

    struct VMTestCase {
        input: String,
        expected_top: Option<Object>,
    }

    #[test]
    fn test_gobal_let_statement() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "let one = 1; one".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "let one = 1; let two = 2; one + two".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "let one = 1; let two = one + one; one + two".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_conditionals() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: "if (true) { 10 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: "if (true) { 10 } else { 20 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(20)),
                input: "if (false) { 10 } else { 20 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: "if (1) { 10 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: "if (1 < 2) { 10 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: "if (1 < 2) { 10 } else { 20 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(20)),
                input: "if (1 > 2) { 10 } else { 20 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "if (1 > 2) { 10 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "if (false) { 10 }".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(20)),
                input: "if ((if (false) { 10 })) { 10 } else { 20 }".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_integer_arithmetic() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "1".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(2)),
                input: "2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "1 + 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(-1)),
                input: "1 - 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(2)),
                input: "1 * 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(2)),
                input: "4 / 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(55)),
                input: "50 / 2 * 2 + 10 - 5".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: "5 + 5 + 5 + 5 - 10".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(32)),
                input: "2 * 2 * 2 * 2 * 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(20)),
                input: "5 * 2 + 10".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(25)),
                input: "5 + 2 * 10".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(60)),
                input: "5 * (2 + 10)".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(-5)),
                input: "-5".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(-10)),
                input: "-10".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(0)),
                input: "-50 + 100 + -50".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(50)),
                input: "(5 + 10 * 2 + 15 / 3) * 2 + -10".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_string_expression() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::String("monkey".to_string())),
                input: "\"monkey\"".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::String("monkey".to_string())),
                input: "\"mon\" + \"key\"".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::String("monkeybanana".to_string())),
                input: "\"mon\" + \"key\"+ \"banana\"".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_hash_literal() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Hash(HashMap::new())),
                input: "{}".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Hash(
                    [
                        (Object::Int(1), Object::Int(2)),
                        (Object::Int(2), Object::Int(3)),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                )),
                input: "{1: 2, 2: 3}".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Hash(
                    [
                        (Object::Int(2), Object::Int(4)),
                        (Object::Int(6), Object::Int(16)),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                )),
                input: "{1 + 1: 2*2, 3+3: 4*4}".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    fn test_index_expression() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "[1,2,3][1]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "[1,2,3][0 + 2]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "[[1,1,1]][0][0]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "[][0]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "[1][-1]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "[1][1]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "{1: 1, 2:2}[1]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(2)),
                input: "{1: 1, 2:2}[2]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "{1: 1, 2:2}[0]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "{}[0]".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_array_literals() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Array(vec![])),
                input: "[]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                ])),
                input: "[1,2,3]".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![
                    Object::Int(3),
                    Object::Int(12),
                    Object::Int(11),
                ])),
                input: "[1+2,3*4,5+6]".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_function_calls() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(15)),
                input: "let fun = fn() { 5 + 10; }; fun();".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "let one = fn() { 1; }; let two = fn() {2}; one() + two();".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "let a = fn() { 1; }; let b = fn() {a() + 1}; let c = fn() {b() + 1}; c();"
                    .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(5)),
                input: "let fun = fn() { return 5;10; }; fun();".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(5)),
                input: "let fun = fn() { return 5; return 10; }; fun();".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "let fun = fn() { }; fun();".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "let fun = fn() { }; let funner = fn() {fun()}; fun(); funner();"
                    .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "let fun = fn() { 1; }; let funner = fn() {fun}; funner()();".to_string(),
            },
            // VMTestCase {
            //     expected_top: Some(Object::Array(vec![
            //         Object::Int(1),
            //         Object::Int(2),
            //         Object::Int(3),
            //     ])),
            //     input: "[1,2,3]".to_string(),
            // },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_bool_expression() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "true".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "1 < 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "1 > 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "1 < 1".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "1 > 1".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "1 == 1".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "1 != 1".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "1 == 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "1 != 2".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "true == true".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "false == false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "true == false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "true != false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "false != true".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "(1 < 2) == true".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "(1 < 2) == false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "(1 > 2) == true".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "(1 > 2) == false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "!true".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "!false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "!5".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "!!true".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(false)),
                input: "!!false".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "!!5".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Bool(true)),
                input: "!(if (false) {5;})".to_string(),
            },
        ];

        run_vm_test(tests);
    }

    fn parse(input: String) -> parser::ast::Program {
        let l = lexer::Lexer::new(input);
        let mut p = parser::Parser::new(l);
        p.parse_program()
    }

    fn run_vm_test(tests: Vec<VMTestCase>) {
        for test in tests {
            let prog = parse(test.input);
            let mut st = SymbolTable::new();
            let mut consts: Vec<Object> = vec![];
            let mut c = Compiler::new_with_state(&mut st, &mut consts);
            let compile_result = c.compile(prog);
            assert!(compile_result.is_ok());

            let mut globals: Vec<Object> = vec![];
            let mut vmm = VM::new_with_global_store(c.bytecode(), &mut globals);
            let result = vmm.run();
            assert!(!result.is_err());
            let stack_elem = vmm.last_popped();

            assert_eq!(stack_elem, test.expected_top);
        }
    }
}
