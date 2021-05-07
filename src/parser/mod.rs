mod ast;
use crate::token::Token;
use crate::{lexer::lexer, token};

use self::ast::*;

pub struct Parser {
    l: lexer,
    cur_token: Token,
    peek_token: Token,
    errors: Vec<String>,
}

fn variant_eq<T>(a: &T, b: &T) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

impl Parser {
    fn new(lex: lexer) -> Self {
        let mut p = Self {
            l: lex,
            cur_token: Token::EOF,
            peek_token: Token::EOF,
            errors: vec![],
        };
        p.next_token();
        p.next_token();
        return p;
    }

    fn next_token(&mut self) {
        self.cur_token = self.peek_token.clone();
        self.peek_token = self.l.next_token();
    }

    pub fn parse_program(&mut self) -> Program {
        let mut p = vec![];
        let mut cur_tok = self.cur_token.clone();

        while cur_tok != Token::EOF {
            let stmt = self.parse_statement();
            if stmt.is_some() {
                p.push(stmt.unwrap());
            }
            self.next_token();
            cur_tok = self.cur_token.clone();
        }
        return p;
    }

    fn parse_statement(&mut self) -> Option<Statement> {
        match self.cur_token {
            Token::LET => Some(self.parse_let_statement()),
            Token::RETURN => Some(self.parse_return_statement()),
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let_statement(&mut self) -> Statement {
        //
        if !self.expect_peek(Token::IDENT(String::from(""))) {
            return Statement::Blank;
        }
        let ident = Ident::from(self.cur_token.clone());
        if !self.expect_peek(Token::ASSIGN) {
            return Statement::Blank;
        }

        // Todo parse expression
        while self.cur_token != Token::SEMICOLON {
            self.next_token();
        }
        Statement::Let(ident, Expression::Blank)
    }

    fn parse_return_statement(&mut self) -> Statement {
        self.next_token();
        // Todo parse expression
        while self.cur_token != Token::SEMICOLON {
            self.next_token();
        }
        Statement::Return(Expression::Blank)
    }

    fn parse_expression_statement(&mut self) -> Option<Statement> {
        let expression = self.parse_expression(Precedence::Lowest);
        if self.peek_token == Token::SEMICOLON {
            self.next_token();
        }
        if expression.is_none() {
            // TODO: errors
            // self.errors
            //     .push(String::from("could not parse expression statement"));
            return None;
        }
        return Some(Statement::Expression(expression.unwrap()));
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Option<Expression> {
        let mut left_expression = match self.cur_token {
            Token::IDENT(_) => self.parse_ident_expression(),
            Token::INT(_) => self.parse_int_expression(),
            // Token::String(_) => self.parse_string_expr(),
            // Token::Bool(_) => self.parse_bool_expr(),
            // Token::Lbracket => self.parse_array_expr(),
            // Token::Lbrace => self.parse_hash_expr(),
            Token::BANG | Token::MINUS => self.parse_prefix_expression(),
            // Token::Lparen => self.parse_grouped_expr(),
            // Token::If => self.parse_if_expr(),
            // Token::Func => self.parse_func_expr(),
            _ => {
                // self.error_no_prefix_Parser();
                None
            }
        };

        while self.peek_token != Token::SEMICOLON
            && precedence < precedence_of(self.peek_token.clone())
        {
            match self.peek_token {
                Token::PLUS
                | Token::MINUS
                | Token::ASTERISK
                | Token::SLASH
                | Token::EQ
                | Token::NE
                | Token::GTE
                | Token::GT
                | Token::LT
                | Token::LTE => {
                    self.next_token();
                    left_expression = self.parse_infix_expression(left_expression.unwrap());
                }
                _ => return left_expression,
            }
        }
        left_expression
    }

    fn parse_infix_expression(&mut self, left: Expression) -> Option<Expression> {
        let infix = match self.cur_token {
            Token::PLUS => Infix::Plus,
            Token::MINUS => Infix::Minus,
            Token::ASTERISK => Infix::Multiply,
            Token::SLASH => Infix::Divide,
            Token::GT => Infix::Gt,
            Token::GTE => Infix::Gte,
            Token::LT => Infix::Lt,
            Token::LTE => Infix::Lte,
            Token::EQ => Infix::Eq,
            Token::NE => Infix::Ne,
            _ => return None,
        };

        let p = precedence_of(self.cur_token.clone());
        self.next_token();

        match self.parse_expression(p) {
            Some(expression) => Some(Expression::Infix(
                infix,
                Box::new(left),
                Box::new(expression),
            )),
            None => None,
        }
    }

    fn parse_ident(&mut self) -> Option<Ident> {
        match self.cur_token {
            Token::IDENT(ref mut ident) => Some(Ident(ident.clone())),
            _ => None,
        }
    }

    fn parse_ident_expression(&mut self) -> Option<Expression> {
        match self.parse_ident() {
            Some(ident) => Some(Expression::Ident(ident)),
            None => None,
        }
    }

    fn parse_int_expression(&mut self) -> Option<Expression> {
        match self.cur_token {
            Token::INT(i) => Some(Expression::Literal(Literal::Int(i))),
            _ => None,
        }
    }

    fn parse_prefix_expression(&mut self) -> Option<Expression> {
        let prefix = match self.cur_token {
            Token::BANG => Prefix::Bang,
            Token::MINUS => Prefix::Minus,
            _ => return None,
        };
        self.next_token();
        match self.parse_expression(Precedence::Prefix) {
            Some(expression) => Some(Expression::Prefix(prefix, Box::new(expression))),
            _ => None,
        }
    }

    fn expect_peek(&mut self, t: Token) -> bool {
        if variant_eq(&t, &self.peek_token) {
            self.next_token();
            return true;
        } else {
            self.peek_error(t);
            return false;
        }
    }

    pub fn errors(&self) -> Vec<String> {
        return self.errors.clone();
    }

    fn peek_error(&mut self, t: Token) {
        self.errors.push(format!(
            "expected token to be {:?}, but got {:?}",
            t, self.peek_token
        ))
    }
}

#[cfg(test)]
mod tests {
    use ast::{Expression, Prefix, Statement};

    use crate::lexer;

    use super::*;

    fn check_errors(p: Parser) {
        let errs = p.errors;
        let empty_vec: Vec<String> = vec![];
        assert_eq!(errs, empty_vec);
    }

    #[test]
    fn test_let_statements() {
        let input = r#"
        let x = 5;
        let y = 10;
        let foobar = 838383;
        "#;

        let l = lexer::lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 3, "expected 3 statements");
        let expected: [Statement; 3] = [
            Statement::Let(
                Ident(String::from("x")),
                Expression::Blank,
                // Expression::Literal(Literal::Int(5)),
            ),
            Statement::Let(
                Ident(String::from("y")),
                Expression::Blank,
                // Expression::Literal(Literal::Int(10)),
            ),
            Statement::Let(
                Ident(String::from("foobar")),
                Expression::Blank,
                // Expression::Literal(Literal::Int(838383)),
            ),
        ];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_let_statements_error() {
        let input = r#"
        let x = 5;
        let 10;
        let  = 838383;
        "#;

        let l = lexer::lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        let errs = p.errors();
        // print!("{:?}", errs);
        assert_eq!(errs.len(), 2, "We're expecting two errors");
    }

    #[test]
    fn test_return_statements() {
        let input = r#"
      return 5;
      return 10;
      return 838383;
      "#;

        let l = lexer::lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 3, "expected 3 statements");
        let expected: [Statement; 3] = [
            Statement::Return(
                Expression::Blank,
                // Expression::Literal(Literal::Int(5)),
            ),
            Statement::Return(
                Expression::Blank,
                // Expression::Literal(Literal::Int(10)),
            ),
            Statement::Return(
                Expression::Blank,
                // Expression::Literal(Literal::Int(838383)),
            ),
        ];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_print() {
        assert_eq!(
            Statement::Let(Ident(String::from("something")), Expression::Blank).to_string(),
            String::from("let something = ;")
        );
        assert_eq!(
            Statement::Let(
                Ident(String::from("something")),
                Expression::Literal(Literal::Int(1234))
            )
            .to_string(),
            String::from("let something = 1234;")
        );

        assert_eq!(
            Statement::Let(
                Ident(String::from("something")),
                Expression::Literal(Literal::Bool(true))
            )
            .to_string(),
            String::from("let something = true;")
        );

        assert_eq!(
            Statement::Let(
                Ident(String::from("something")),
                Expression::Literal(Literal::String(String::from("asdf")))
            )
            .to_string(),
            String::from("let something = asdf;")
        );
    }

    #[test]
    fn test_expression_statements() {
        let input = r#"
      foobar;
      5;
      "#;

        let l = lexer::lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 2, "expected 2 statements");
        let expected: [Statement; 2] = [
            Statement::Expression(Expression::Ident(Ident(String::from("foobar")))),
            Statement::Expression(Expression::Literal(Literal::Int(5))),
        ];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_prefix_statements() {
        let input = r#"
      -foobar;
      !5;
      "#;

        let l = lexer::lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 2, "expected 2 statements");
        let expected: [Statement; 2] = [
            Statement::Expression(Expression::Prefix(
                Prefix::Minus,
                Box::new(Expression::Ident(Ident(String::from("foobar")))),
            )),
            Statement::Expression(Expression::Prefix(
                Prefix::Bang,
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
        ];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_infix_statements() {
        let input = r#"
      5 * 5;
      5 + 5;
      5 / 5;
      5 - 5;
      5 > 5;
      5 < 5;
      5 <= 5;
      5 >= 5;
      5 == 5;
      5 != 5;
      "#;

        let l = lexer::lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 10, "expected 10 statements");
        let expected: [Statement; 10] = [
            Statement::Expression(Expression::Infix(
                Infix::Multiply,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Plus,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Divide,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Minus,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Gt,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Lt,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Lte,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Gte,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Eq,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Ne,
                Box::new(Expression::Literal(Literal::Int(5))),
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
        ];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_infix_statements_precedence_strings() {
        let input = r#"
        -a * b;
        !-a;
        a + b + c;
        a + b - c;
        a * b * c;
        a * b / c;
        a + b / c;
        a + b * c + d / e - f;
        3 + 4; -5 * 5;
        5 > 4 == 3 < 4;
        5 < 4 != 3 > 4;
        3 + 4 * 5 == 3 * 1 + 4 * 5;
      "#;

        let l = lexer::lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        let expected: [&str; 13] = [
            "((-a) * b)",
            "(!(-a))",
            "((a + b) + c)",
            "((a + b) - c)",
            "((a * b) * c)",
            "((a * b) / c)",
            "(a + (b / c))",
            "(((a + (b * c)) + (d / e)) - f)",
            "(3 + 4)",
            "((-5) * 5)",
            "((5 > 4) == (3 < 4))",
            "((5 < 4) != (3 > 4))",
            "((3 + (4 * 5)) == ((3 * 1) + (4 * 5)))",
        ];
        assert_eq!(
            program.len(),
            expected.len(),
            "expected length and recieved lengths differ"
        );
        for i in 0..program.len() {
            // println!("{}", program[i].to_string())
            assert_eq!(program[i].to_string(), expected[i], "{}", i);
        }
    }
}
