use crate::{
    code::{InstructionList, Instructions, Opcode},
    compiler::Bytecode,
    evaluator::object::Object,
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
                Opcode::Add => {
                    let left = match self.pop() {
                        Object::Int(int) => int,
                        _ => return Err(VMError::Reason("unexpected type to add".to_string())),
                    };
                    let right = match self.pop() {
                        Object::Int(int) => int,
                        _ => return Err(VMError::Reason("unexpected type to add".to_string())),
                    };

                    self.push(Object::Int(left + right))?;
                }
            }

            ip += 1;
        }

        Ok(())
    }

    pub fn push(&mut self, obj: Object) -> Result<(), VMError> {
        if self.sp >= self.stack.capacity() {
            return Err(VMError::Reason("Stack overflow".to_string()));
        }

        self.stack.push(obj);
        self.sp += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Object {
        self.sp -= 1;
        self.stack
            .pop()
            .expect("Expected something to be on the stack")
    }
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    use crate::code::*;
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
            let stack_elem = vmm.stack_top();

            assert_eq!(stack_elem, test.expected_top);
        }
    }
}
