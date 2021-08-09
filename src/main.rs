#![allow(dead_code)]
mod lexer;
mod parser;
use std::io::{self, BufRead};
mod code;
mod compiler;
mod evaluator;
mod token;
mod vm;
// use token::Token;
use evaluator::builtins::new_builtins;
use evaluator::env;
use evaluator::object::Object;
// use std::cell::RefCell;
// use std::rc::Rc;

use io::Write;

use compiler::symbol_table::{self, SymbolTable};
use vm::VM;

fn main() {
    let mut env = env::Env::from(new_builtins());

    env.set(
        String::from("puts"),
        &Object::Builtin(-1, |args| {
            for arg in args {
                println!("{}", arg);
            }
            Object::Null
        }),
    );

    // let mut the_evaluator = evaluator::Evaluator::new(Rc::new(RefCell::new(env)));

    let mut constants: Vec<Object> = vec![];
    let mut globals: Vec<Object> = Vec::with_capacity(vm::GLOBALS_SIZE);
    let mut st: symbol_table::SymbolTable = SymbolTable::new();

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
        let mut comp = compiler::Compiler::new_with_state(&mut st, &mut constants);
        if let Err(e) = comp.compile(program) {
            println!("Compilation error: {:?}", e);
        }

        let code = comp.bytecode();
        // constants = code.constants.clone();
        // // st = comp.symbol_table.clone();
        // println!("{:?}", comp.bytecode());

        let mut machine = VM::new_with_global_store(code, &mut globals);
        if let Err(e) = machine.run() {
            println!("Error Executing Code: {:?}", e);
        }
        // globals = machine.globals.clone();
        if let Some(result) = machine.last_popped() {
            println!("{}", result);
        } else {
            println!(" :C ");
        }
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
