use crate::code::*;
use crate::parser::ast::Statement;
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
            instructions: vec![],
            constants: vec![],
        };
    }

    pub fn compile(&self, s: Vec<Statement>) -> Result<(), CompileError> {
        Ok(())
    }

    pub fn bytecode(&self) -> Bytecode {
        return Bytecode {
            instructions: self.instructions.clone(),
            constants: self.constants.clone(),
        };
    }
}

pub struct Bytecode {
    instructions: Vec<u8>,
    constants: Vec<Object>,
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
        let tests: Vec<CompilerTestCase> = vec![CompilerTestCase {
            input: "1 + 2".to_string(),
            expected_constants: vec![Object::Int(1), Object::Int(2)],
            expected_instructions: vec![
                make(Opcode::Constant, vec![0]).unwrap(),
                make(Opcode::Constant, vec![1]).unwrap(),
            ],
        }];

        run_compiler_test(tests);
    }

    fn run_compiler_test(tests: Vec<CompilerTestCase>) {
        for test in tests {
            let program = parse(test.input);
            let c = Compiler::new();
            let compile_result = c.compile(program);
            assert!(compile_result.is_ok());

            let bytecode = c.bytecode();

            let instruction_result =
                test_instructions(test.expected_instructions, bytecode.instructions);
            assert!(instruction_result.is_ok());

            let constant_result = test_constants(test.expected_constants, bytecode.constants);
            assert!(constant_result.is_ok());
        }
    }

    fn test_instructions(expected: Vec<Vec<u8>>, got: Vec<u8>) -> Result<(), CompileError> {
        let concatted = concat_instructions(expected);

        if got.len() != concatted.len() {
            assert_eq!(concatted.len(), got.len());
        }

        for (i, ins) in concatted.iter().enumerate() {
            assert_eq!(got.get(i).unwrap(), ins);
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
            for b in e {
                out.push(b);
            }
        }
        return out;
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
