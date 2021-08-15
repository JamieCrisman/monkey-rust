pub mod symbol_table;

use crate::code::*;
use crate::parser::ast::{BlockStatement, Expression, Ident, Infix, Literal, Prefix, Statement};
use crate::Object;
use std::cell::RefCell;
use std::rc::Rc;

use self::symbol_table::SymbolTable;

#[derive(Clone)]
struct EmittedInstruction {
    op: Opcode,
    position: usize,
}

pub struct CompilationScope {
    instructions: Instructions,
    last_instruction: Option<EmittedInstruction>,
    previous_instruction: Option<EmittedInstruction>,
}

pub struct Compiler {
    // instructions: Instructions,
    pub constants: Rc<RefCell<Vec<Object>>>,
    // last_instruction: Option<EmittedInstruction>,
    // previous_instruction: Option<EmittedInstruction>,
    scope_index: usize,
    scopes: Vec<CompilationScope>,
    pub symbol_table: Rc<RefCell<SymbolTable>>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum CompileError {
    Reason(String),
}

impl Compiler {
    // pub fn new() -> Self {
    //     return ;
    // }

    pub fn new_with_state(
        st: Rc<RefCell<SymbolTable>>,
        constants: Rc<RefCell<Vec<Object>>>,
    ) -> Self {
        Compiler {
            // instructions: Instructions { data: vec![] },
            constants: constants,
            // last_instruction: None,
            // previous_instruction: None,
            symbol_table: st,
            scope_index: 0,
            scopes: vec![CompilationScope {
                instructions: Instructions { data: vec![] },
                last_instruction: None,
                previous_instruction: None,
            }],
        }
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
            Statement::Let(l, e) => self.compile_let(l, e),
            Statement::Return(e) => self.compile_return(e),
            _ => Err(CompileError::Reason("Not Implemented".to_string())),
        }
    }

    fn compile_return(&mut self, e: Expression) -> Result<(), CompileError> {
        self.compile_expression(e)?;
        self.emit(Opcode::ReturnValue, None);
        Ok(())
    }

    fn current_instructions(&mut self) -> Option<Instructions> {
        let instructions = match self.scopes.get(self.scope_index) {
            Some(s) => s.instructions.clone(),
            None => return None,
        };
        return Some(instructions);
    }

    fn compile_expression(&mut self, exp: Expression) -> Result<(), CompileError> {
        match exp {
            Expression::Blank => Ok(()),
            Expression::Ident(ident) => {
                let symbol = match self.symbol_table.borrow_mut().resolve(ident.0.clone()) {
                    None => {
                        return Err(CompileError::Reason(format!(
                            "undefined variable: {}",
                            ident.0
                        )))
                    }
                    Some(val) => val,
                };
                let val = Some(vec![symbol.index as i32]);
                match symbol.scope {
                    symbol_table::SymbolScope::Global => self.emit(Opcode::GetGlobal, val),
                    symbol_table::SymbolScope::Local => self.emit(Opcode::GetLocal, val),
                    symbol_table::SymbolScope::BuiltIn => self.emit(Opcode::BuiltinFunc, val),
                    symbol_table::SymbolScope::Free => self.emit(Opcode::GetFree, val),
                    symbol_table::SymbolScope::Function => self.emit(Opcode::CurrentClosure, None),
                };
                Ok(())
            }
            Expression::Infix(i, exp_a, exp_b) => self.compile_infix(i, exp_a, exp_b),
            Expression::Prefix(p, exp) => self.compile_prefix(p, exp),
            Expression::Literal(literal) => self.compile_literal(literal),
            Expression::If {
                condition,
                consequence,
                alternative,
            } => self.compile_if(condition, consequence, alternative),
            Expression::Index(expr, ind_expr) => self.compile_index(expr, ind_expr),
            Expression::Func { params, body, name } => self.compile_function(params, body, name),
            Expression::Call { args, func } => self.compile_call(args, func),
            _ => Err(CompileError::Reason("Not Implemented".to_string())),
        }
    }

    fn compile_call(
        &mut self,
        args: Vec<Expression>,
        func: Box<Expression>,
    ) -> Result<(), CompileError> {
        self.compile_expression(*func)?;
        let len = args.len() as i32;
        for a in args {
            self.compile_expression(a)?;
        }
        self.emit(Opcode::Call, Some(vec![len]));
        Ok(())
    }

    fn compile_function(
        &mut self,
        params: Vec<Ident>,
        body: BlockStatement,
        name: String,
    ) -> Result<(), CompileError> {
        self.enter_scope();
        if name.len() != 0 {
            self.symbol_table.borrow_mut().define_function(name.clone());
        }
        let param_len = params.len() as i32;
        for p in params {
            self.symbol_table.borrow_mut().define(p.0.as_str());
        }

        self.compile(body)?;
        if self.last_instruction_is(Opcode::Pop) {
            self.replace_last_pop_with_return();
        }
        if !self.last_instruction_is(Opcode::ReturnValue) {
            self.emit(Opcode::Return, None);
        }
        let num_locals = self.symbol_table.borrow().num_definitions;
        let free_symbols = self.symbol_table.borrow().free_symbols.clone();
        let instr = self.leave_scope();

        for sym in free_symbols.iter() {
            self.load_symbol(sym.clone());
        }

        let compiled_fn = Object::CompiledFunction {
            instructions: instr,
            num_locals: num_locals as i32,
            num_parameters: param_len,
        };
        let constant_val = Some(vec![
            self.add_constant(compiled_fn) as i32,
            free_symbols.len() as i32,
        ]);
        self.emit(Opcode::Closure, constant_val);
        Ok(())
    }

    fn load_symbol(&mut self, symbol: symbol_table::Symbol) {
        match symbol.scope {
            symbol_table::SymbolScope::Global => {
                self.emit(Opcode::GetGlobal, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Local => {
                self.emit(Opcode::GetLocal, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::BuiltIn => {
                self.emit(Opcode::BuiltinFunc, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Free => {
                self.emit(Opcode::GetFree, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Function => self.emit(Opcode::CurrentClosure, None),
        };
    }

    fn replace_last_pop_with_return(&mut self) {
        let last_pos = self.scopes[self.scope_index]
            .last_instruction
            .as_ref()
            .unwrap()
            .position;
        self.replace_instruction(last_pos, make(Opcode::ReturnValue, None).unwrap().data);
        self.scopes[self.scope_index]
            .last_instruction
            .as_mut()
            .unwrap()
            .op = Opcode::ReturnValue;
    }

    fn compile_let(&mut self, l: Ident, e: Expression) -> Result<(), CompileError> {
        let symbol = self.symbol_table.borrow_mut().define(l.0.as_str());
        self.compile_expression(e)?;
        match symbol.scope {
            symbol_table::SymbolScope::Global => {
                self.emit(Opcode::SetGlobal, Some(vec![symbol.index as i32]))
            }
            symbol_table::SymbolScope::Local => {
                self.emit(Opcode::SetLocal, Some(vec![symbol.index as i32]))
            }
            _ => return Err(CompileError::Reason("Cannot set a builtin".to_string())),
        };
        Ok(())
    }

    fn compile_index(
        &mut self,
        expr: Box<Expression>,
        ind_expr: Box<Expression>,
    ) -> Result<(), CompileError> {
        self.compile_expression(*expr)?;
        self.compile_expression(*ind_expr)?;
        self.emit(Opcode::Index, None);
        Ok(())
    }

    fn compile_if(
        &mut self,
        condition: Box<Expression>,
        consequence: BlockStatement,
        alternative: Option<BlockStatement>,
    ) -> Result<(), CompileError> {
        self.compile_expression(*condition)?;
        // this gets properly set later (back patching)
        let jump_not_truthy_pos = self.emit(Opcode::JumpNotTruthy, Some(vec![9999]));
        self.compile(consequence)?;

        if self.last_instruction_is(Opcode::Pop) {
            self.remove_last_pop();
        }
        // to be backpatched
        let jump_pos = self.emit(Opcode::Jump, Some(vec![9999]));
        let after_consequence_pos = self.scopes[self.scope_index].instructions.data.len();
        self.change_operand(
            jump_not_truthy_pos,
            Some(vec![after_consequence_pos as i32]),
        );

        if alternative.is_none() {
            self.emit(Opcode::Null, None);
        } else {
            self.compile(alternative.unwrap())?;

            if self.last_instruction_is(Opcode::Pop) {
                self.remove_last_pop();
            }
        }
        let after_alternative_pos = self.scopes[self.scope_index].instructions.data.len();
        self.change_operand(jump_pos, Some(vec![after_alternative_pos as i32]));

        Ok(())
    }

    fn last_instruction_is(&mut self, op: Opcode) -> bool {
        self.scopes
            .get(self.scope_index)
            .unwrap()
            .last_instruction
            .is_some()
            && self
                .scopes
                .get(self.scope_index)
                .unwrap()
                .last_instruction
                .as_ref()
                .unwrap()
                .op
                == op
    }

    fn remove_last_pop(&mut self) {
        // self.instructions
        let pos = self.scopes[self.scope_index]
            .last_instruction
            .as_ref()
            .unwrap()
            .position;
        while self.scopes[self.scope_index].instructions.data.len() > pos {
            self.scopes[self.scope_index].instructions.data.pop();
        }
        self.scopes[self.scope_index].last_instruction =
            self.scopes[self.scope_index].previous_instruction.clone();
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
            Literal::String(s) => {
                let operand = Some(vec![self.add_constant(Object::String(s)) as i32]);
                self.emit(Opcode::Constant, operand)
            }
            Literal::Array(elements) => {
                let size = Some(vec![elements.len() as i32]);
                for element in elements {
                    // println!("element: {:?}", element);
                    self.compile_expression(element)?;
                }
                self.emit(Opcode::Array, size);
                return Ok(());
            }
            Literal::Hash(hash) => {
                // TODO:: sort by hash key?

                for (k, v) in hash.iter() {
                    self.compile_expression(k.clone())?;
                    self.compile_expression(v.clone())?;
                }
                self.emit(Opcode::Hash, Some(vec![(hash.len() * 2) as i32]));

                return Ok(());
            }
            _ => return Err(CompileError::Reason("Not Implemented".to_string())),
        };
        Ok(())
    }

    fn change_operand(&mut self, opcode_pos: usize, operands: Option<Vec<i32>>) {
        let op = Opcode::from(
            self.current_instructions()
                .expect("expected instructions to exist")
                .data[opcode_pos],
        );
        let new_instruction = make(op, operands).expect("expected result");
        self.replace_instruction(opcode_pos, new_instruction.data);
    }

    fn replace_instruction(&mut self, pos: usize, new_instruction: Vec<u8>) {
        for (i, ins) in new_instruction.iter().enumerate() {
            self.scopes[self.scope_index].instructions.data[pos + i] = *ins;
        }
    }

    fn add_constant(&mut self, obj: Object) -> usize {
        self.constants.borrow_mut().push(obj);
        return self.constants.borrow().len() - 1;
    }

    fn emit(&mut self, op: Opcode, operands: Option<Vec<i32>>) -> usize {
        let ins = make(op.clone(), operands);
        let pos = self.add_instruction(ins.expect("wanted valid instructions"));
        self.set_last_instruction(op, pos);
        return pos;
    }

    fn set_last_instruction(&mut self, op: Opcode, pos: usize) {
        let prev = self
            .scopes
            .get(self.scope_index)
            .unwrap()
            .last_instruction
            .clone();
        let last = EmittedInstruction { op, position: pos };
        self.scopes[self.scope_index].previous_instruction = prev;
        self.scopes[self.scope_index].last_instruction = Some(last);
    }

    fn add_instruction(&mut self, ins: Instructions) -> usize {
        let pos = match self.current_instructions() {
            Some(instr) => instr.data.len(),
            None => 0,
        };
        let mut ins_copy = ins.clone();
        self.scopes[self.scope_index]
            .instructions
            .data
            .append(&mut ins_copy.data);
        return pos;
    }

    pub fn bytecode(&self) -> Bytecode {
        return Bytecode {
            instructions: self.scopes[self.scope_index].instructions.clone(),
            constants: self.constants.borrow().clone(),
        };
    }

    fn enter_scope(&mut self) {
        let scope = CompilationScope {
            instructions: Instructions { data: vec![] },
            last_instruction: None,
            previous_instruction: None,
        };
        self.scopes.push(scope);
        self.scope_index += 1;
        let new_table = Rc::new(RefCell::new(SymbolTable::new_with_outer(
            self.symbol_table.to_owned(),
        )));
        self.symbol_table = new_table;
    }

    fn leave_scope(&mut self) -> Instructions {
        let results = self.current_instructions();

        self.scopes.pop();
        self.scope_index -= 1;

        // let _inner = self.symbol_table.;
        let outer = self
            .symbol_table
            .borrow()
            .outer
            .as_ref()
            .unwrap()
            .to_owned();
        self.symbol_table = outer;
        // TODO: drop?
        return results.unwrap();
    }
}

#[derive(Debug)]
pub struct Bytecode {
    pub instructions: Instructions,
    pub constants: Vec<Object>,
}

#[cfg(test)]
mod tests {

    use std::vec;

    // use crate::evaluator::builtins::new_builtins;
    use crate::compiler::*;
    use crate::lexer;
    use crate::parser;

    struct CompilerTestCase {
        input: String,
        expected_constants: Vec<Object>,
        expected_instructions: Vec<Instructions>,
    }

    fn test_compiler_scopes() {
        let st = Rc::new(RefCell::new(SymbolTable::new()));
        let constants: Rc<RefCell<Vec<Object>>> = Rc::new(RefCell::new(vec![]));
        let mut c = Compiler::new_with_state(st, constants);
        c.emit(Opcode::Multiply, None);
        c.enter_scope();
        assert_eq!(1, c.scope_index);
        c.emit(Opcode::Subtract, None);
        assert_eq!(
            1,
            c.scopes.get(c.scope_index).unwrap().instructions.data.len()
        );

        assert!(c.symbol_table.as_ref().borrow().outer.is_some());
        // assert_ne!(c.symbol_table, None);

        let last = c
            .scopes
            .get(c.scope_index)
            .unwrap()
            .last_instruction
            .clone()
            .unwrap();
        assert_eq!(Opcode::Subtract, last.op);
        c.leave_scope();
        assert_eq!(0, c.scope_index);
        assert!(c.symbol_table.as_ref().borrow().outer.is_none());
        c.emit(Opcode::Add, None);
        assert_eq!(
            2,
            c.scopes.get(c.scope_index).unwrap().instructions.data.len()
        );
        let last2 = c
            .scopes
            .get(c.scope_index)
            .unwrap()
            .last_instruction
            .clone()
            .unwrap();
        assert_eq!(Opcode::Add, last2.op);
        let previous = c
            .scopes
            .get(c.scope_index)
            .unwrap()
            .previous_instruction
            .clone()
            .unwrap();
        assert_eq!(Opcode::Multiply, previous.op);
    }

    fn parse(input: String) -> parser::ast::Program {
        let l = lexer::Lexer::new(input);
        let mut p = parser::Parser::new(l);
        p.parse_program()
    }

    #[test]
    fn test_builtin_functions() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "len([]);push([],1);".to_string(),
                expected_constants: vec![Object::Int(1)],
                expected_instructions: vec![
                    make(Opcode::BuiltinFunc, Some(vec![0])).unwrap(),
                    make(Opcode::Array, Some(vec![0])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                    make(Opcode::BuiltinFunc, Some(vec![4])).unwrap(),
                    make(Opcode::Array, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Call, Some(vec![2])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() {len([]);}".to_string(),
                expected_constants: vec![Object::CompiledFunction {
                    instructions: concat_instructions(vec![
                        make(Opcode::BuiltinFunc, Some(vec![0])).unwrap(),
                        make(Opcode::Array, Some(vec![0])).unwrap(),
                        make(Opcode::Call, Some(vec![1])).unwrap(),
                        make(Opcode::ReturnValue, None).unwrap(),
                    ]),
                    num_locals: 0,
                    num_parameters: 0,
                }],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "len([1,2,3])".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2), Object::Int(3)],
                expected_instructions: vec![
                    make(Opcode::BuiltinFunc, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Array, Some(vec![3])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_closure() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "fn(a) { fn(b) { a + b }}".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetFree, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Closure, Some(vec![0, 1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn(a) { fn(b) { fn(c) {a + b + c }}}".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetFree, Some(vec![0])).unwrap(),
                            make(Opcode::GetFree, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetFree, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Closure, Some(vec![0, 2])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Closure, Some(vec![1, 1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let g = 55; fn() { let a = 66; fn() { let b = 77; fn() { let c = 88; g + a + b + c; }}}".to_string(),
                expected_constants: vec![
                    Object::Int(55),
                    Object::Int(66),
                    Object::Int(77),
                    Object::Int(88),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![3])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                            make(Opcode::GetFree, Some(vec![0])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::GetFree, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 0,
                    },
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![2])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetFree, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Closure, Some(vec![4, 2])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 0,
                    },
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Closure, Some(vec![5, 1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Closure, Some(vec![6, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            // CompilerTestCase {
            //     input: "fn() {len([]);}".to_string(),
            //     expected_constants: vec![Object::CompiledFunction {
            //         instructions: concat_instructions(vec![
            //             make(Opcode::BuiltinFunc, Some(vec![0])).unwrap(),
            //             make(Opcode::Array, Some(vec![0])).unwrap(),
            //             make(Opcode::Call, Some(vec![1])).unwrap(),
            //             make(Opcode::ReturnValue, None).unwrap(),
            //         ]),
            //         num_locals: 0,
            //         num_parameters: 0,
            //     }],
            //     expected_instructions: vec![
            //         make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
            //         make(Opcode::Pop, None).unwrap(),
            //     ],
            // },
            // CompilerTestCase {
            //     input: "len([1,2,3])".to_string(),
            //     expected_constants: vec![Object::Int(1), Object::Int(2), Object::Int(3)],
            //     expected_instructions: vec![
            //         make(Opcode::BuiltinFunc, Some(vec![0])).unwrap(),
            //         make(Opcode::Constant, Some(vec![0])).unwrap(),
            //         make(Opcode::Constant, Some(vec![1])).unwrap(),
            //         make(Opcode::Constant, Some(vec![2])).unwrap(),
            //         make(Opcode::Array, Some(vec![3])).unwrap(),
            //         make(Opcode::Call, Some(vec![1])).unwrap(),
            //         make(Opcode::Pop, None).unwrap(),
            //     ],
            // },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_functions() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "fn() { return 5 + 10 }".to_string(),
                expected_constants: vec![
                    Object::Int(5),
                    Object::Int(10),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { 5 + 10 }".to_string(),
                expected_constants: vec![
                    Object::Int(5),
                    Object::Int(10),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { 1; 2 }".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Pop, None).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { }".to_string(),
                expected_constants: vec![Object::CompiledFunction {
                    instructions: concat_instructions(vec![make(Opcode::Return, None).unwrap()]),
                    num_locals: 0,
                    num_parameters: 0,
                }],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_recursive() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "let countDown = fn(x) { countDown(x-1) }; countDown(1);".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::CurrentClosure, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Subtract, None).unwrap(),
                            make(Opcode::Call, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::Int(1),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let wrapper = fn() {let countDown = fn(x) { countDown(x-1) }; countDown(1);}; wrapper();".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::CurrentClosure, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::Subtract, None).unwrap(),
                            make(Opcode::Call, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::Int(1),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Closure, Some(vec![1,0])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![2])).unwrap(),
                            make(Opcode::Call, Some(vec![1])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![3, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Call, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_function_calls() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "fn() { 24 }()".to_string(),
                expected_constants: vec![
                    Object::Int(24),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::Call, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let noArg = fn() { 24 };noArg();".to_string(),
                expected_constants: vec![
                    Object::Int(24),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Call, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let aArg = fn(a) { };aArg(24);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(
                            vec![make(Opcode::Return, None).unwrap()],
                        ),
                        num_parameters: 1,
                        num_locals: 1,
                    },
                    Object::Int(24),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let someArgs = fn(a,b,c) { };someArgs(24,25,26);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(
                            vec![make(Opcode::Return, None).unwrap()],
                        ),
                        num_locals: 3,
                        num_parameters: 3,
                    },
                    Object::Int(24),
                    Object::Int(25),
                    Object::Int(26),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Call, Some(vec![3])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let someArg = fn(a) { a };someArg(24);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 1,
                    },
                    Object::Int(24),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Call, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let someArg = fn(a, b, c) { a;b;c };someArg(24,25,26);".to_string(),
                expected_constants: vec![
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Pop, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![1])).unwrap(),
                            make(Opcode::Pop, None).unwrap(),
                            make(Opcode::GetLocal, Some(vec![2])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 3,
                        num_parameters: 3,
                    },
                    Object::Int(24),
                    Object::Int(25),
                    Object::Int(26),
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![0, 0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Call, Some(vec![3])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_index_expression() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "[1, 2, 3][1+1]".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(1),
                    Object::Int(1),
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Array, Some(vec![3])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Constant, Some(vec![4])).unwrap(),
                    make(Opcode::Add, None).unwrap(),
                    make(Opcode::Index, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "{1: 2}[2-1]".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(2),
                    Object::Int(1),
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Hash, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Subtract, None).unwrap(),
                    make(Opcode::Index, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_array_literals() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "[]".to_string(),
                expected_constants: vec![],
                expected_instructions: vec![
                    make(Opcode::Array, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "[1, 2, 3]".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2), Object::Int(3)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Array, Some(vec![3])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "[1 + 2, 3 - 4, 5 * 6]".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                    Object::Int(5),
                    Object::Int(6),
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Add, None).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Subtract, None).unwrap(),
                    make(Opcode::Constant, Some(vec![4])).unwrap(),
                    make(Opcode::Constant, Some(vec![5])).unwrap(),
                    make(Opcode::Multiply, None).unwrap(),
                    make(Opcode::Array, Some(vec![3])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_string_expression() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "\"monkey\"".to_string(),
                expected_constants: vec![Object::String("monkey".to_string())],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "\"mon\" + \"key\"".to_string(),
                expected_constants: vec![
                    Object::String("mon".to_string()),
                    Object::String("key".to_string()),
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Add, None).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_hash_literal() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "{}".to_string(),
                expected_constants: vec![],
                expected_instructions: vec![
                    make(Opcode::Hash, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "{1:2,3:4,5:6}".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                    Object::Int(5),
                    Object::Int(6),
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Constant, Some(vec![4])).unwrap(),
                    make(Opcode::Constant, Some(vec![5])).unwrap(),
                    make(Opcode::Hash, Some(vec![6])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "{1:2+3,4:5*6}".to_string(),
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                    Object::Int(5),
                    Object::Int(6),
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    make(Opcode::Add, None).unwrap(),
                    make(Opcode::Constant, Some(vec![3])).unwrap(),
                    make(Opcode::Constant, Some(vec![4])).unwrap(),
                    make(Opcode::Constant, Some(vec![5])).unwrap(),
                    make(Opcode::Multiply, None).unwrap(),
                    make(Opcode::Hash, Some(vec![4])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_let_scopes() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "let one = 100; fn() { one; };".to_string(),
                expected_constants: vec![
                    Object::Int(100),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 0,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { let one = 100; one; };".to_string(),
                expected_constants: vec![
                    Object::Int(100),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 1,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![1, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "fn() { let one = 100; let two = 200; one+two };".to_string(),
                expected_constants: vec![
                    Object::Int(100),
                    Object::Int(200),
                    Object::CompiledFunction {
                        instructions: concat_instructions(vec![
                            make(Opcode::Constant, Some(vec![0])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::Constant, Some(vec![1])).unwrap(),
                            make(Opcode::SetLocal, Some(vec![1])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![0])).unwrap(),
                            make(Opcode::GetLocal, Some(vec![1])).unwrap(),
                            make(Opcode::Add, None).unwrap(),
                            make(Opcode::ReturnValue, None).unwrap(),
                        ]),
                        num_locals: 2,
                        num_parameters: 0,
                    },
                ],
                expected_instructions: vec![
                    make(Opcode::Closure, Some(vec![2, 0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            // CompilerTestCase {
            //     input: "let one = 1; one;".to_string(),
            //     expected_constants: vec![Object::Int(1)],
            //     expected_instructions: vec![
            //         make(Opcode::Constant, Some(vec![0])).unwrap(),
            //         make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::Pop, None).unwrap(),
            //     ],
            // },
            // CompilerTestCase {
            //     input: "let one = 1; let two = one; two;".to_string(),
            //     expected_constants: vec![Object::Int(1)],
            //     expected_instructions: vec![
            //         make(Opcode::Constant, Some(vec![0])).unwrap(),
            //         make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
            //         make(Opcode::SetGlobal, Some(vec![1])).unwrap(),
            //         make(Opcode::GetGlobal, Some(vec![1])).unwrap(),
            //         make(Opcode::Pop, None).unwrap(),
            //     ],
            // },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_global_let() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "let one = 1; let two = 2;".to_string(),
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![1])).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let one = 1; one;".to_string(),
                expected_constants: vec![Object::Int(1)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "let one = 1; let two = one; two;".to_string(),
                expected_constants: vec![Object::Int(1)],
                expected_instructions: vec![
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![0])).unwrap(),
                    make(Opcode::SetGlobal, Some(vec![1])).unwrap(),
                    make(Opcode::GetGlobal, Some(vec![1])).unwrap(),
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
    }

    #[test]
    fn test_conditionals() {
        let tests: Vec<CompilerTestCase> = vec![
            CompilerTestCase {
                input: "if (true) { 10 } else { 20 }; 3333;".to_string(),
                expected_constants: vec![Object::Int(10), Object::Int(20), Object::Int(3333)],
                expected_instructions: vec![
                    // 00
                    make(Opcode::True, None).unwrap(),
                    // 01
                    make(Opcode::JumpNotTruthy, Some(vec![10])).unwrap(),
                    // 04
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    // 07
                    make(Opcode::Jump, Some(vec![13])).unwrap(),
                    // 10
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    // 13
                    make(Opcode::Pop, None).unwrap(),
                    // 14
                    make(Opcode::Constant, Some(vec![2])).unwrap(),
                    // 17
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
            CompilerTestCase {
                input: "if (true) { 10 }; 3333;".to_string(),
                expected_constants: vec![Object::Int(10), Object::Int(3333)],
                expected_instructions: vec![
                    // 00
                    make(Opcode::True, None).unwrap(),
                    // 01
                    make(Opcode::JumpNotTruthy, Some(vec![10])).unwrap(),
                    // 04
                    make(Opcode::Constant, Some(vec![0])).unwrap(),
                    // 07
                    make(Opcode::Jump, Some(vec![11])).unwrap(),
                    // 10
                    make(Opcode::Null, None).unwrap(),
                    // 11
                    make(Opcode::Pop, None).unwrap(),
                    // 12
                    make(Opcode::Constant, Some(vec![1])).unwrap(),
                    // 15
                    make(Opcode::Pop, None).unwrap(),
                ],
            },
        ];

        run_compiler_test(tests);
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
            println!("testing {}", test.input);
            let program = parse(test.input.clone());
            let st = Rc::new(RefCell::new(SymbolTable::new_with_builtins()));
            let constants: Rc<RefCell<Vec<Object>>> = Rc::new(RefCell::new(vec![]));
            let mut c = Compiler::new_with_state(st, constants);
            // println!("{:?}", program);
            let compile_result = c.compile(program);
            assert!(
                compile_result.is_ok(),
                "{:?}",
                compile_result
                    .err()
                    .unwrap_or(CompileError::Reason("uh... not sure".to_string()))
            );

            let bytecode = c.bytecode();
            // println!("{:?}", test.input);
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
        println!("len {}: {:?}", got.data.len(), got.data);
        println!("len {}: {:?}", concatted.data.len(), concatted.data);
        if got.data.len() != concatted.data.len() {
            assert_eq!(concatted.data.len(), got.data.len());
        }

        println!("{:?}", got.data);
        println!("{:?}", concatted.data);
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
                    _ => return Err(CompileError::Reason("wrong comparison types".to_string())),
                },
                Object::String(s) => match got.get(i).unwrap() {
                    Object::String(s2) => assert_eq!(s, s2),
                    _ => return Err(CompileError::Reason("wrong comparison types".to_string())),
                },
                Object::CompiledFunction {
                    instructions: Instructions { data },
                    num_locals,
                    num_parameters,
                } => match got.get(i).unwrap() {
                    Object::CompiledFunction {
                        instructions: Instructions { data: data2 },
                        num_locals: locals2,
                        num_parameters: params2,
                    } => {
                        assert_eq!(data, data2);
                        assert_eq!(num_locals, locals2);
                        assert_eq!(num_parameters, params2);
                    }
                    _ => return Err(CompileError::Reason("wrong comparison types".to_string())),
                },
                _ => {}
            }
        }

        Ok(())
    }
}
