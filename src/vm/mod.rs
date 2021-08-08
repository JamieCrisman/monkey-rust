use crate::{
    code::{Instructions, Opcode},
    compiler::Bytecode,
    evaluator::object::{Object, ObjectType},
};

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum VMError {
    Reason(String),
}

const DEFAULT_STACK_SIZE: usize = 2048;

pub struct VM {
    constants: Vec<Object>,
    instructions: Instructions,
    stack: Vec<Object>,
    sp: usize,
    // stack_size: i32,
}

impl VM {
    pub fn new(bytecode: Bytecode) -> Self {
        return Self {
            instructions: bytecode.instructions.clone(),
            constants: bytecode.constants.clone(),
            sp: 0,
            stack: Vec::with_capacity(DEFAULT_STACK_SIZE),
        };
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
        let mut ip: usize = 0;
        while ip < self.instructions.data.len() {
            let op = Opcode::from(*self.instructions.data.get(ip).expect("expected byte"));
            match op {
                Opcode::Constant => {
                    let buff = [
                        *self.instructions.data.get(ip + 1).expect("expected byte"),
                        *self.instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let const_index = u16::from_be_bytes(buff);
                    ip += *op.width().unwrap().get(0).unwrap() as usize;
                    self.push(self.constants.get(const_index as usize).unwrap().clone())?;
                }
                Opcode::Add | Opcode::Divide | Opcode::Multiply | Opcode::Subtract => {
                    self.execute_binary_operation(op)?;
                }
                Opcode::Pop => {
                    self.pop();
                }
            }

            ip += 1;
        }

        Ok(())
    }

    fn execute_binary_operation(&mut self, op: Opcode) -> Result<(), VMError> {
        let right = self.pop();
        let left = self.pop();

        match (left.object_type(), right.object_type()) {
            (ObjectType::Int, ObjectType::Int) => {
                self.execute_binary_integer_operation(op, left, right)?;
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
    use crate::compiler::*;
    use crate::evaluator::object::*;
    use crate::lexer;
    use crate::parser;
    use crate::vm::VM;

    struct VMTestCase {
        input: String,
        expected_top: Option<Object>,
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
            let mut c = Compiler::new();
            let compile_result = c.compile(prog);
            assert!(compile_result.is_ok());

            let mut vmm = VM::new(c.bytecode());
            let result = vmm.run();
            assert!(!result.is_err());
            let stack_elem = vmm.last_popped();

            assert_eq!(stack_elem, test.expected_top);
        }
    }
}
