use crate::code::*;
use crate::parser::ast::{Expression, Infix, Literal, Prefix, Statement};
use crate::Object;

pub struct Compiler {
    instructions: Instructions,
    constants: Vec<Object>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum CompileError {
    Reason(String),
}

impl Compiler {
    pub fn new() -> Self {
        return Compiler {
            instructions: Instructions { data: vec![] },
            constants: vec![],
        };
    }

    pub fn compile(&mut self, program: Vec<Statement>) -> Result<(), CompileError> {
        for s in program {
            if let Err(e) = self.compile_statement(s) {
                return Err(e);
            }
        }
        Ok(())
    }

    fn compile_statement(&mut self, statement: Statement) -> Result<(), CompileError> {
        match statement {
            Statement::Blank => Ok(()),
            Statement::Expression(e) => {
                self.compile_expression(e)?;
                self.emit(Opcode::Pop, None);
                Ok(())
            }
            // Statement::Let(l, e) => self.compile_let(l, e),
            // Statement::Return(e) => self.compile_return(r),
            _ => Err(CompileError::Reason("Not Implemented".to_string())),
        }
    }

    fn compile_expression(&mut self, exp: Expression) -> Result<(), CompileError> {
        match exp {
            Expression::Blank => Ok(()),
            Expression::Infix(i, exp_a, exp_b) => self.compile_infix(i, exp_a, exp_b),
            Expression::Prefix(p, exp) => self.compile_prefix(p, exp),
            Expression::Literal(literal) => self.compile_literal(literal),
            _ => Err(CompileError::Reason("Not Implemented".to_string())),
        }
    }

    fn compile_prefix(&mut self, prefix: Prefix, exp: Box<Expression>) -> Result<(), CompileError> {
        self.compile_expression(*exp)?;
        match prefix {
            Prefix::Bang => self.emit(Opcode::Bang, None),
            Prefix::Minus => self.emit(Opcode::Minus, None),
        };
        Ok(())
    }

    fn compile_infix(
        &mut self,
        infix: Infix,
        exp_a: Box<Expression>,
        exp_b: Box<Expression>,
    ) -> Result<(), CompileError> {
        if infix == Infix::Lt {
            self.compile_expression(*exp_b)?;
            self.compile_expression(*exp_a)?;
            self.emit(Opcode::GreaterThan, None);
            return Ok(());
        }

        if let Err(e) = self.compile_expression(*exp_a) {
            return Err(e);
        }
        if let Err(e) = self.compile_expression(*exp_b) {
            return Err(e);
        }

        match infix {
            Infix::Plus => self.emit(Opcode::Add, None),
            Infix::Divide => self.emit(Opcode::Divide, None),
            Infix::Multiply => self.emit(Opcode::Multiply, None),
            Infix::Minus => self.emit(Opcode::Subtract, None),
            Infix::Eq => self.emit(Opcode::Equal, None),
            Infix::Ne => self.emit(Opcode::NotEqual, None),
            Infix::Gt => self.emit(Opcode::GreaterThan, None),
            _ => return Err(CompileError::Reason("Not Implemented".to_string())),
        };
        Ok(())
    }

    fn compile_literal(&mut self, l: Literal) -> Result<(), CompileError> {
        match l {
            Literal::Int(int) => {
                let ind = self.add_constant(Object::Int(int)) as i32;
                self.emit(Opcode::Constant, Some(vec![ind]))
            }
            Literal::Bool(b) => {
                if b {
                    self.emit(Opcode::True, None)
                } else {
                    self.emit(Opcode::False, None)
                }
            }
            _ => return Err(CompileError::Reason("Not Implemented".to_string())),
        };
        Ok(())
    }

    fn add_constant(&mut self, obj: Object) -> usize {
        self.constants.push(obj);
        return self.constants.len() - 1;
    }

    fn emit(&mut self, op: Opcode, operands: Option<Vec<i32>>) -> usize {
        let ins = make(op, operands);
        return self.add_instruction(ins.expect("wanted valid instructions"));
    }

    fn add_instruction(&mut self, ins: Instructions) -> usize {
        let pos = self.instructions.data.len();
        let mut ins_copy = ins.clone();
        self.instructions.data.append(&mut ins_copy.data);
        return pos;
    }

    pub fn bytecode(&self) -> Bytecode {
        return Bytecode {
            instructions: self.instructions.clone(),
            constants: self.constants.clone(),
        };
    }
}

#[derive(Debug)]
pub struct Bytecode {
    pub instructions: Instructions,
    pub constants: Vec<Object>,
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    use crate::compiler::*;
    use crate::lexer;
    use crate::parser;

    struct CompilerTestCase {
        input: String,
        expected_constants: Vec<Object>,
        expected_instructions: Vec<Instructions>,
    }

    fn parse(input: String) -> parser::ast::Program {
        let l = lexer::Lexer::new(input);
        let mut p = parser::Parser::new(l);
        p.parse_program()
    }

    #[test]
    fn test_integer_arithmetic() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "1 + 2".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Add, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1;2".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 - 2".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Subtract, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 * 2".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Multiply, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "2 / 1".to_string(),
                expected_constants: vec![Object::Int(2), Object::Int(1)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Divide, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "-1".to_string(),
                expected_constants: vec![Object::Int(1)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Minus, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_bool_arithmetic() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "true".to_string(),
                expected_constants: vec![],
                expected_instructions: vec![
                    make(Opcode::True, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "!true".to_string(),
                expected_constants: vec![],
                expected_instructions: vec![
                    make(Opcode::True, None).unwrap(),
                    make(Opcode::Bang, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "false".to_string(),
                expected_constants: vec![],
                expected_instructions: vec![
                    make(Opcode::False, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 > 2".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::GreaterThan, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 < 2".to_string(),
                expected_constants: vec![Object::Int(2), Object::Int(1)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::GreaterThan, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 == 2".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Equal, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "1 != 2".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::NotEqual, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "true == false".to_string(),
                expected_constants: vec![],
                expected_instructions: vec![
                    make(Opcode::True, None).unwrap(),
                    make(Opcode::False, None).unwrap(),
                    make(Opcode::Equal, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "true != false".to_string(),
                expected_constants: vec![],
                expected_instructions: vec![
                    make(Opcode::True, None).unwrap(),
                    make(Opcode::False, None).unwrap(),
                    make(Opcode::NotEqual, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    fn run_compiler_test(tests: Vec<CompilerTestCase>) {
        for test in tests {
            let program = parse(test.input);
            let mut c = Compiler::new();
            // println!("{:?}", program);
            let compile_result = c.compile(program);
            assert!(compile_result.is_ok());

            let bytecode = c.bytecode();
            // println!("{:?}", bytecode);
            let instruction_result =
                test_instructions(test.expected_instructions, bytecode.instructions);
            assert!(instruction_result.is_ok());

            let constant_result = test_constants(test.expected_constants, bytecode.constants);
            assert!(constant_result.is_ok());
        }
    }

    fn test_instructions(
        expected: Vec<Instructions>,
        got: Instructions,
    ) -> Result<(), CompileError> {
        let concatted = concat_instructions(expected);

        if got.data.len() != concatted.data.len() {
            assert_eq!(concatted.data.len(), got.data.len());
        }

        // println!("{:?}", result_list);
        // println!("{:?}", concatted);
        for (i, ins) in concatted.data.iter().enumerate() {
            assert_eq!(got.data.get(i).unwrap(), ins);
            // if got.get(i).unwrap() != ins {
            //     return Err(CompileError::Reason(format!(
            //         "wrong instruction at {}, got: {:?} wanted: {:?}",
            //         i, concatted, got
            //     )));
            // }
        }

        Ok(())
    }

    fn concat_instructions(expected: Vec<Instructions>) -> Instructions {
        let mut out: Vec<u8> = vec![];
        for e in expected {
            for b in e.data {
                out.push(b);
            }
        }
        return Instructions { data: out };
    }

    fn test_constants(expected: Vec<Object>, got: Vec<Object>) -> Result<(), CompileError> {
        assert_eq!(expected.len(), got.len());

        for (i, c) in expected.iter().enumerate() {
            match c {
                Object::Int(v) => match got.get(i).unwrap() {
                    Object::Int(v2) => assert_eq!(v, v2),
                    _ => {}
                },
                _ => {}
            }
        }

        Ok(())
    }
}
