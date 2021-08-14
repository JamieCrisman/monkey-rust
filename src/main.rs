#![allow(dead_code)]
mod lexer;
mod parser;
use std::io::{self, BufRead};
mod builtins;
mod code;
mod compiler;
mod evaluator;
mod token;
mod vm;
// use token::Token;
// use evaluator::builtins::new_builtins;
// use evaluator::env;
use evaluator::object::Object;
use std::cell::RefCell;
use std::rc::Rc;

use io::Write;

use compiler::symbol_table::{self, SymbolTable};
use vm::VM;

fn main() {
    // let mut env = env::Env::from(new_builtins());

    // let mut the_evaluator = evaluator::Evaluator::new(Rc::new(RefCell::new(env)));

    let mut constants: Rc<RefCell<Vec<Object>>> = Rc::new(RefCell::new(vec![]));
    let mut globals: Rc<RefCell<Vec<Object>>> =
        Rc::new(RefCell::new(Vec::with_capacity(vm::GLOBALS_SIZE)));
    let mut st: Rc<RefCell<symbol_table::SymbolTable>> =
        Rc::new(RefCell::new(SymbolTable::new_with_builtins()));

    loop {
        print!(">> ");
        let mut line = String::new();
        let stdin = io::stdin();
        io::stdout().flush().unwrap();
        stdin.lock().read_line(&mut line).unwrap();
        let l = lexer::Lexer::new(line);
        let mut p = parser::Parser::new(l);
        let program = p.parse_program();
        let errs = p.errors();
        if errs.len() != 0 {
            for e in errs {
                println!("{}", e);
            }
        }
        // println!("{:?}", st);
        let mut comp = compiler::Compiler::new_with_state(st, constants);
        if let Err(e) = comp.compile(program) {
            println!("Compilation error: {:?}", e);
        }

        let code = comp.bytecode();
        st = comp.symbol_table;
        constants = comp.constants;

        // constants = code.constants.clone();
        // st = comp.symbol_table.clone();
        // println!("{:?}", comp.symbol_table);

        let mut machine = VM::new_with_global_store(code, globals);
        if let Err(e) = machine.run() {
            println!("Error Executing Code: {:?}", e);
        }
        // globals = machine.globals.clone();
        if let Some(result) = machine.last_popped() {
            println!("{}", result);
        } else {
            println!(" :C ");
        }
        globals = machine.globals;
        // if let Some(result) = the_evaluator.eval(program) {
        //     println!("{}\n", result);
        // }
        // let mut tok = l.next_token();
        // while tok != Token::EOF {
        //     // println!("{:?}", tok);
        //     tok = l.next_token();
        // }
        // print!("{}", line)
    }
}
