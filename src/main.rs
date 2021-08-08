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

use crate::vm::VM;

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
        let mut comp = compiler::Compiler::new();
        if let Err(e) = comp.compile(program) {
            println!("Compilation error: {:?}", e);
        }
        let mut machine = VM::new(comp.bytecode());
        if let Err(e) = machine.run() {
            println!("Error Executing Code: {:?}", e);
        }

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
