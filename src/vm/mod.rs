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
    instructions: Instructions,
    stack: Vec<Object>,
    sp: usize,
    pub globals: Vec<Object>,
    // stack_size: i32,
}

impl VM {
    pub fn new(bytecode: Bytecode) -> Self {
        return Self {
            instructions: bytecode.instructions.clone(),
            constants: bytecode.constants.clone(),
            sp: 0,
            stack: Vec::with_capacity(DEFAULT_STACK_SIZE),
            globals: Vec::with_capacity(GLOBALS_SIZE),
        };
    }

    pub fn new_with_global_store(bytecode: Bytecode, g: Vec<Object>) -> Self {
        let mut result = Self::new(bytecode);
        result.globals = g;
        result
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
                    ip += op.operand_width() as usize;
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
                        *self.instructions.data.get(ip + 1).expect("expected byte"),
                        *self.instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let jump_target = u16::from_be_bytes(buff);
                    ip = jump_target as usize - 1;
                }
                Opcode::JumpNotTruthy => {
                    let buff = [
                        *self.instructions.data.get(ip + 1).expect("expected byte"),
                        *self.instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let jump_target = u16::from_be_bytes(buff);
                    ip += 2;
                    let condition = self.pop();
                    if !is_truthy(condition) {
                        ip = jump_target as usize - 1;
                    }
                }
                Opcode::Pop => {
                    self.pop();
                }
                Opcode::Null => self.push(NULL)?,
                Opcode::GetGlobal => {
                    let buff = [
                        *self.instructions.data.get(ip + 1).expect("expected byte"),
                        *self.instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let global_index = u16::from_be_bytes(buff);
                    ip += 2;
                    let val = (*self.globals.get(global_index as usize).unwrap()).clone();
                    self.push(val)?;
                }
                Opcode::SetGlobal => {
                    let buff = [
                        *self.instructions.data.get(ip + 1).expect("expected byte"),
                        *self.instructions.data.get(ip + 2).expect("expected byte"),
                    ];
                    let global_index = u16::from_be_bytes(buff);
                    ip += 2;
                    let pop = self.pop();
                    self.globals.insert(global_index as usize, pop);
                }
            }

            ip += 1;
        }

        Ok(())
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
