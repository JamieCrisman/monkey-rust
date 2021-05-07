mod lexer;
mod parser;
use std::io::{self, BufRead};
mod token;
use token::Token;

use io::Write;

fn main() {
    loop {
        print!(">> ");
        let mut line = String::new();
        let stdin = io::stdin();
        io::stdout().flush().unwrap();
        stdin.lock().read_line(&mut line).unwrap();
        let mut l = lexer::lexer::new(line);
        let mut tok = l.next_token();
        while tok != Token::EOF {
            // println!("{:?}", tok);
            tok = l.next_token();
        }
        // print!("{}", line)
    }
}
