mod frame;
use core::num;
use std::cell::RefCell;
use std::rc::Rc;
use std::{borrow::Borrow, collections::HashMap, ops::Deref};

use crate::builtins::{self, BuiltInFunction};
use crate::evaluator::object::BuiltInFunc;
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

pub struct VM {
    constants: Vec<Object>,
    // instructions: Instructions,
    stack: Vec<Object>,
    last_popped: Option<Object>,
    sp: usize,
    pub globals: Rc<RefCell<Vec<Object>>>,
    // stack_size: i32,
    frames: Vec<Frame>,
    frames_index: i32,
    builtins: Vec<BuiltInFunction>,
}

impl VM {
    pub fn new_with_global_store(bytecode: Bytecode, g: Rc<RefCell<Vec<Object>>>) -> Self {
        let frames = vec![Frame::new(bytecode.instructions.clone(), 0, 0, vec![], 0)];
        // frames[0].ip += 1;
        Self {
            // instructions: bytecode.instructions.clone(),
            constants: bytecode.constants.clone(),
            last_popped: None,
            sp: 0,
            // stack: vec![Object::Null; DEFAULT_STACK_SIZE],
            stack: Vec::with_capacity(DEFAULT_STACK_SIZE),
            globals: g,
            frames,
            frames_index: 1,
            builtins: builtins::new_builtins(),
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
        // let fin = self.frames_index;
        // println!(
        //     "ind {} cur {:?}",
        //     fin,
        //     self.current_frame() // self.current_frame()
        //                          //     .instructions()
        //                          //     .expect("expected instructions")
        //                          //     .data
        //                          //     .len()
        // );
        while self.current_frame().ip
            < (self
                .current_frame()
                .instructions()
                .expect("expected instructions")
                .data
                .len() as i64
                - 1)
        {
            let init_ip = self.current_frame().ip;
            self.set_ip((init_ip + 1) as i64);
            let ip = self.current_frame().ip as usize;
            // let instr = self.current_frame().instructions().unwrap();
            // println!("cur ip {} instr: {:?}", ip, instr);
            // self.set_ip(ip as i64);
            let cur_instructions = self
                .current_frame()
                .instructions()
                .expect("expected instructions");
            let op = Opcode::from(*cur_instructions.data.get(ip).expect("expected byte"));
            // println!(" ------- got opcode: {:?} ip: {} sp: {}", op, ip, self.sp);

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
                    // println!("sp {} stack {:?}", self.sp, self.stack);
                    // println!("const (ind: {}) {:?}", const_index, self.constants);

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
                    let val: Object = self
                        .globals
                        .as_ref()
                        .borrow()
                        .get(global_index as usize)
                        .unwrap()
                        .clone();
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
                    self.globals.borrow_mut().insert(global_index as usize, pop);
                }
                Opcode::Array => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let element_count = u16::from_be_bytes(buff);
                    self.set_ip((ip + 2) as i64);
                    let array = self.build_array(self.sp - element_count as usize, self.sp);
                    self.sp = self.sp - element_count as usize;
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
                    let arg_count = cur_instructions
                        .data
                        .get(ip + 1)
                        .expect("expected byte")
                        .clone() as i64;
                    self.set_ip((ip + 1) as i64);
                    self.execute_call_function(arg_count)?;
                }
                Opcode::Return => {
                    let f = self.pop_frame();
                    self.pop();
                    self.sp = (f.base_pointer - 1) as usize;

                    self.push(Object::Null)?
                }
                Opcode::ReturnValue => {
                    let ret_val = self.pop();
                    let f = self.pop_frame();
                    self.sp = (f.base_pointer - 1) as usize;
                    // self.pop();
                    self.push(ret_val)?;
                    // cur_instructions = self
                    //     .current_frame()
                    //     .instructions()
                    //     .expect("expected instructions");
                }
                Opcode::GetLocal => {
                    let local_index = cur_instructions
                        .data
                        .get(ip + 1)
                        .expect("expected byte")
                        .clone() as i64;
                    self.set_ip((ip + 1) as i64);
                    let base_pointer = self.current_frame().base_pointer;
                    let val = self.stack[(base_pointer + local_index) as usize].clone();
                    self.push(val)?;
                }
                Opcode::SetLocal => {
                    let local_index = cur_instructions
                        .data
                        .get(ip + 1)
                        .expect("expected byte")
                        .clone() as i64;
                    self.set_ip((ip + 1) as i64);
                    let base_pointer = self.current_frame().base_pointer;
                    // println!(
                    //     " -- local_index: {} base_pointer: {}",
                    //     local_index, base_pointer
                    // );
                    self.stack[(base_pointer + local_index) as usize] = self.pop();
                }
                Opcode::BuiltinFunc => {
                    let built_index = cur_instructions
                        .data
                        .get(ip + 1)
                        .expect("expected byte")
                        .clone() as i64;
                    self.set_ip((ip + 1) as i64);

                    let def = self
                        .builtins
                        .get(built_index as usize)
                        .unwrap()
                        .func
                        .clone();
                    self.push(def)?;
                }
                Opcode::Closure => {
                    let buff = [
                        *cur_instructions.data.get(ip + 1).expect("expected byte"),
                        *cur_instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let const_index = u16::from_be_bytes(buff);
                    let num_free = cur_instructions
                        .data
                        .get(ip + 3)
                        .expect("expected byte")
                        .clone() as i64;
                    self.set_ip((ip + 3) as i64);
                    self.push_closure(const_index, num_free)?;
                }
                Opcode::GetFree => {
                    let free_ind = cur_instructions
                        .data
                        .get(ip + 1)
                        .expect("expected byte")
                        .clone() as i64;
                    self.set_ip((ip + 1) as i64);
                    let var = self
                        .current_frame()
                        .free
                        .get(free_ind as usize)
                        .expect("expected a variable to exist")
                        .clone();
                    self.push(var)?;
                }
                Opcode::CurrentClosure => {
                    let cur_frame = self.current_frame();
                    let obj = Object::Closure {
                        Free: cur_frame.free.clone(),
                        Fn: Box::new(Object::CompiledFunction {
                            instructions: cur_frame.instr.clone(),
                            num_locals: cur_frame.num_locals,
                            num_parameters: cur_frame.num_args as i32,
                        }),
                    };
                    self.push(obj)?;
                }
            }
        }

        Ok(())
    }

    fn push_closure(&mut self, const_index: u16, num_free: i64) -> Result<(), VMError> {
        let constant = self.constants.get(const_index as usize);
        let function = if let Some(obj) = constant {
            match obj.clone() {
                Object::CompiledFunction {
                    instructions: _,
                    num_locals: _,
                    num_parameters: _,
                } => obj,
                _ => return Err(VMError::Reason("Expected compiled function".to_string())),
            }
        } else {
            return Err(VMError::Reason("Expected constant".to_string()));
        };
        let mut free: Vec<Object> = vec![];
        for i in 0..num_free {
            free.push(self.stack[(self.sp - num_free as usize) + (i as usize)].clone());
        }
        let cl = Object::Closure {
            Fn: Box::new(function.to_owned()),
            Free: free,
        };
        self.sp = self.sp - num_free as usize;
        self.push(cl)?;

        Ok(())
    }

    fn execute_call_function(&mut self, args: i64) -> Result<(), VMError> {
        match self.stack[self.sp - 1 - (args as usize)].clone() {
            Object::CompiledFunction {
                instructions,
                num_locals,
                num_parameters,
            } => {
                if num_parameters != args as i32 {
                    return Err(VMError::Reason(format!(
                        "wrong number of arguments: want {} but got {}",
                        num_parameters, args,
                    )));
                }
                self.execute_compiled_function(args, instructions, num_locals)?;
            }
            Object::Builtin(num_parameters, builtin_func) => {
                if num_parameters != -1 && num_parameters != args as i32 {
                    return Err(VMError::Reason(format!(
                        "wrong number of arguments: want {} but got {}",
                        num_parameters, args,
                    )));
                }
                self.execute_builtin(builtin_func, args)?;
            }
            Object::Closure { Fn, Free } => {
                let (instructions, num_locals, num_parameters) = match *Fn {
                    Object::CompiledFunction {
                        instructions,
                        num_locals,
                        num_parameters,
                    } => (instructions, num_locals, num_parameters),
                    something_else => {
                        return Err(VMError::Reason(format!(
                            "expected function, but got {:?}({})",
                            something_else.object_type(),
                            something_else
                        )))
                    }
                };
                if num_parameters != -1 && num_parameters != args as i32 {
                    return Err(VMError::Reason(format!(
                        "wrong number of arguments: want {} but got {}",
                        num_parameters, args,
                    )));
                }
                self.execute_closure(num_parameters as i64, instructions, num_locals, Free)?;
            }
            something_else => {
                return Err(VMError::Reason(format!(
                    "expected function, but got {:?}({})",
                    something_else.object_type(),
                    something_else
                )));
            }
        };

        Ok(())
    }

    fn execute_closure(
        &mut self,
        num_args: i64,
        instructions: Instructions,
        num_locals: i32,
        free: Vec<Object>,
    ) -> Result<(), VMError> {
        let new_frame = Frame::new(
            instructions,
            num_locals,
            self.sp as i64 - num_args,
            free,
            num_args,
        );
        let bp = new_frame.base_pointer;
        self.push_frame(new_frame);
        for _i in 0..num_locals {
            self.push(Object::Null)?;
        }
        self.sp = (bp + (num_locals as i64)) as usize;
        Ok(())
    }

    fn execute_builtin(&mut self, func: BuiltInFunc, num_args: i64) -> Result<(), VMError> {
        let args = self
            .stack
            .get(self.sp - (num_args as usize)..self.sp)
            .unwrap()
            .to_vec();
        let result = func(args);
        self.sp = self.sp - num_args as usize - 1;
        self.push(result)?;
        Ok(())
    }

    fn execute_compiled_function(
        &mut self,
        args: i64,
        instructions: Instructions,
        num_locals: i32,
    ) -> Result<(), VMError> {
        let new_frame = Frame::new(
            instructions,
            num_locals,
            self.sp as i64 - args,
            vec![],
            args,
        );
        let bp = new_frame.base_pointer;
        self.push_frame(new_frame);
        for _i in 0..num_locals {
            self.push(Object::Null)?;
        }
        self.sp = (bp + (num_locals as i64)) as usize;
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
        // let val = self.stack.pop().unwrap();
        let val = self.stack.remove(self.sp - 1);
        self.last_popped = Some(val.clone());
        self.sp -= 1;
        return val;
    }

    pub fn last_popped(&self) -> Option<Object> {
        self.last_popped.clone()
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
    use crate::vm::{VMError, VM};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

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
    fn test_builtin_functions() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(0)),
                input: "len(\"\")".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(4)),
                input: "len(\"four\")".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(11)),
                input: "len(\"hello world\")".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Error(
                    "argument to `len` not supported, got 1".to_string(),
                )),
                input: "len(1)".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Error(
                    "wrong number of arguments: want 1 but got 2".to_string(),
                )),
                input: "len(\"a\", \"b\")".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "len([1,2,3])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(0)),
                input: "len([])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "puts(\"hello\")".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "first([1,2,3])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "first([])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Error(
                    "argument to `first` must be array, got 1".to_string(),
                )),
                input: "first(1)".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "last([1,2,3])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "last([])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Error(
                    "argument to `last` must be array, got 1".to_string(),
                )),
                input: "last(1)".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![Object::Int(2), Object::Int(3)])),
                input: "rest([1,2,3])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Null),
                input: "rest([])".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Array(vec![Object::Int(1)])),
                input: "push([],1)".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Error(
                    "argument to `push` must be array, got 1".to_string(),
                )),
                input: "push(1,1)".to_string(),
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
    fn test_closures() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(99)),
                input: "let cl = fn(a) {fn() {a}}; let closure = cl(99); closure();".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(14)),
                input: r#"
                let newAdderOuter = fn(a, b) {
                    let c = a + b;
                    fn(d) {
                        let e = d + c;
                        fn(f) { e + f;};
                    };
                };
                let newAdderInner = newAdderOuter(1,2);
                let adder = newAdderInner(3);
                adder(8);
                "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(14)),
                input: r#"
                let a = 1;
                let newAdderOuter = fn(b) {
                    fn(c) {
                        fn(d) {a + b + c + d};
                    };
                };
                let newAdderInner = newAdderOuter(2);
                let adder = newAdderInner(3);
                adder(8);
                "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(99)),
                input: r#"
                let newClosure = fn(a,b) {
                    let one = fn() {a;};
                    let two = fn() {b;};
                    fn() { one() + two()};
                };
                let closure = newClosure(9, 90);
                closure();
                "#
                .to_string(),
            },
            // VMTestCase {
            //     expected_top: Some(Object::Int(3)),
            //     input: "let one = fn() { 1; }; let two = fn() {2}; one() + two();".to_string(),
            // },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_recursive() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(0)),
                input: r#"
                let countDown = fn(x) {
                    if (x == 0) {
                        return 0;
                    } else {
                        countDown(x - 1);
                    }
                };
                countDown(1);
                "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(0)),
                input: r#"
                let countDown = fn(x) {
                    if (x == 0) {
                        return 0;
                    } else {
                        countDown(x - 1);
                    }
                };
                let wrapper = fn() {countDown(1)};
                wrapper();
                "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(0)),
                input: r#"
                let wrapper = fn() {
                    let countDown = fn(x) {
                        if (x == 0) {
                            return 0;
                        } else {
                            countDown(x - 1);
                        }
                    };
                    countDown(1);
                };
                wrapper();
                "#
                .to_string(),
            },
            // VMTestCase {
            //     expected_top: Some(Object::Int(14)),
            //     input: r#"
            //     let a = 1;
            //     let newAdderOuter = fn(b) {
            //         fn(c) {
            //             fn(d) {a + b + c + d};
            //         };
            //     };
            //     let newAdderInner = newAdderOuter(2);
            //     let adder = newAdderInner(3);
            //     adder(8);
            //     "#
            //     .to_string(),
            // },
            // VMTestCase {
            //     expected_top: Some(Object::Int(99)),
            //     input: r#"
            //     let newClosure = fn(a,b) {
            //         let one = fn() {a;};
            //         let two = fn() {b;};
            //         fn() { one() + two()};
            //     };
            //     let closure = newClosure(9, 90);
            //     closure();
            //     "#
            //     .to_string(),
            // },
            // VMTestCase {
            //     expected_top: Some(Object::Int(3)),
            //     input: "let one = fn() { 1; }; let two = fn() {2}; one() + two();".to_string(),
            // },
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
    fn test_function_calls_with_bindings() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: "let one = fn() { let one = 1; one }; one();".to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: "let oneAndTwo = fn() { let one = 1; let two = 2; one + two }; oneAndTwo();"
                    .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: r#"
                let oneAndTwo = fn() { let one = 1; let two = 2; one + two };
                let threeAndFour = fn() { let three = 3; let four = 4; three + four };
                oneAndTwo() + threeAndFour();
                "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(150)),
                input: r#"
              let foo = fn() { let foo = 50; foo };
              let alsoFoo = fn() { let foo = 100; foo };
              foo() + alsoFoo();
              "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(97)),
                input: r#"
            let global = 50;
            let minusOne = fn() { let foo = 1; global - foo };
            let minusTwo = fn() { let foo = 2; global - foo };
            minusOne() + minusTwo();
            "#
                .to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_function_calls_with_args_and_bindings() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(4)),
                input: r#"
            let identity = fn(a) { a };
            identity(4);
            "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: r#"
          let sum = fn(a,b) { a+b };
          sum(1,2);
          "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(3)),
                input: r#"
          let sum = fn(a,b) { let c = a+b; c };
          sum(1,2);
          "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: r#"
          let sum = fn(a,b) { let c = a+b; c };
          sum(1,2) + sum(3,4);
          "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(10)),
                input: r#"
          let sum = fn(a,b) { let c = a+b; c };
          let outer = fn() {sum(1,2) + sum(3,4)};
          outer();
          "#
                .to_string(),
            },
            VMTestCase {
                expected_top: Some(Object::Int(50)),
                input: r#"
          let global = 10;
          let sum = fn(a,b) {
              let c = a + b;
              c + global;
          };
          let outer = fn() {
              sum(1,2) + sum(3,4) + global;
          };
          outer() + global;
          "#
                .to_string(),
            },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_function_calls_with_wrong_args() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Error(
                    "wrong number of arguments: want 0 but got 1".to_string(),
                )),
                input: r#"
            fn() { 1; }(1);
            "#
                .to_string(),
            },
            // VMTestCase {
            //     expected_top: Some(Object::Error(
            //         "wrong number of arguments: want 1 but got 0".to_string(),
            //     )),
            //     input: r#"
            //   fn(a){a}();
            //   "#
            //     .to_string(),
            // },
            //     VMTestCase {
            //         expected_top: Some(Object::Error(
            //             "wrong number of arguments: want 2 but got 1".to_string(),
            //         )),
            //         input: r#"
            //   fn(a,b){a+b}(1);
            //   "#
            //         .to_string(),
            //     },
        ];

        run_vm_test(tests);
    }

    #[test]
    fn test_first_class_function_calls() {
        let tests: Vec<VMTestCase> = vec![
            VMTestCase {
                expected_top: Some(Object::Int(1)),
                input: r#"
                let retOneRetter = fn() { let retOne = fn() {1;}; retOne };
                retOneRetter()();
                "#
                .to_string(),
            },
            // VMTestCase {
            //     expected_top: Some(Object::Int(150)),
            //     input: r#"
            //   let foo = fn() { let foo = 50; foo };
            //   let alsoFoo = fn() { let foo = 100; foo };
            //   foo() + alsoFoo();
            //   "#
            //     .to_string(),
            // },
            // VMTestCase {
            //     expected_top: Some(Object::Int(97)),
            //     input: r#"
            // let global = 50;
            // let minusOne = fn() { let foo = 1; global - foo };
            // let minusTwo = fn() { let foo = 2; global - foo };
            // minusOne() + minusTwo();
            // "#
            //     .to_string(),
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
            println!("--- testing: {}", test.input);
            let prog = parse(test.input);
            let st = Rc::new(RefCell::new(SymbolTable::new_with_builtins()));
            let constants: Rc<RefCell<Vec<Object>>> = Rc::new(RefCell::new(vec![]));
            let mut c = Compiler::new_with_state(st, constants);
            let compile_result = c.compile(prog);
            assert!(compile_result.is_ok());

            let globals: Rc<RefCell<Vec<Object>>> = Rc::new(RefCell::new(vec![]));
            let mut vmm = VM::new_with_global_store(c.bytecode(), globals);
            let result = vmm.run();
            match test.expected_top.clone() {
                Some(Object::Error(e)) => {
                    if result.is_err() {
                        match result.err().unwrap() {
                            VMError::Reason(e2) => {
                                assert_eq!(e, e2);
                            }
                        }
                    } else {
                        let stack_elem = vmm.last_popped();
                        assert_eq!(stack_elem, test.expected_top);
                    }
                }
                Some(_) => {
                    assert!(!result.is_err(), "got error: {:?}", result.unwrap_err());
                    let stack_elem = vmm.last_popped();
                    assert_eq!(stack_elem, test.expected_top);
                }
                None => {}
            }
        }
    }
}
