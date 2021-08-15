pub mod ast;
use crate::lexer::Lexer;
use crate::token::Token;

use self::ast::{
    precedence_of, BlockStatement, Expression, Ident, Infix, Literal, Precedence, Prefix, Program,
    Statement,
};

pub struct Parser {
    l: Lexer,
    cur_token: Token,
    peek_token: Token,
    errors: Vec<String>,
}

fn variant_eq<T>(a: &T, b: &T) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

impl Parser {
    pub fn new(lex: Lexer) -> Self {
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
            Token::LET => self.parse_let_statement(),
            Token::RETURN => self.parse_return_statement(),
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_let_statement(&mut self) -> Option<Statement> {
        match self.peek_token {
            Token::IDENT(_) => self.next_token(),
            _ => {
                self.peek_error(Token::IDENT(String::from("any")));
                return None;
            }
        }

        let ident = match self.parse_ident() {
            Some(name) => name,
            None => return None,
        };

        if !self.expect_peek(Token::ASSIGN) {
            return None;
        }

        self.next_token();

        let expression = match self.parse_expression(Precedence::Lowest) {
            Some(expr) => match expr {
                Expression::Func {
                    params,
                    body,
                    name: _,
                } => Some(Expression::Func {
                    params,
                    body,
                    name: ident.0.clone(),
                }),
                _ => Some(expr),
            },
            None => return None,
        };

        if self.peek_token == Token::SEMICOLON {
            self.next_token();
        }

        Some(Statement::Let(ident, expression.unwrap()))
    }

    fn parse_return_statement(&mut self) -> Option<Statement> {
        self.next_token();

        let expression = match self.parse_expression(Precedence::Lowest) {
            Some(expr) => expr,
            None => return None,
        };

        if self.peek_token == Token::SEMICOLON {
            self.next_token();
        }

        Some(Statement::Return(expression))
    }

    fn parse_expression_statement(&mut self) -> Option<Statement> {
        let expression = self.parse_expression(Precedence::Lowest);
        if self.peek_token == Token::SEMICOLON {
            self.next_token();
        }
        if expression.is_none() {
            // TODO:
            // self.errorserrors
            // self.errors
            //     .push(String::from("could not parse expression statement"));
            return None;
        }
        return Some(Statement::Expression(expression.unwrap()));
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Option<Expression> {
        // println!("cur {:?} peek {:?}", self.cur_token, self.peek_token);

        let mut left_expression = match self.cur_token {
            Token::IDENT(_) => self.parse_ident_expression(),
            Token::INT(_) => self.parse_int_expression(),
            Token::STRING(_) => self.parse_string_expression(),
            Token::BOOL(_) => self.parse_bool_expr(),
            Token::LBRACKET => self.parse_array_ident_expr(),
            Token::LBRACE => self.parse_hash_expr(),
            Token::BANG | Token::MINUS => self.parse_prefix_expression(),
            Token::LPAREN => self.parse_grouped_expr(),
            Token::IF => self.parse_if_expression(),
            Token::FUNCTION => self.parse_func_expression(),
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
                Token::LBRACKET => {
                    self.next_token();
                    left_expression = self.parse_index_expr(left_expression.unwrap());
                }
                Token::LPAREN => {
                    self.next_token();
                    left_expression = self.parse_call_expression(left_expression.unwrap());
                }
                _ => return left_expression,
            }
        }
        left_expression
    }

    fn parse_call_expression(&mut self, func: Expression) -> Option<Expression> {
        let args = match self.parse_expression_list(Token::RPAREN) {
            Some(args) => args,
            None => return None,
        };

        Some(Expression::Call {
            func: Box::new(func),
            args,
        })
    }

    fn parse_func_expression(&mut self) -> Option<Expression> {
        if !self.expect_peek(Token::LPAREN) {
            return None;
        }

        let params = match self.parse_func_params() {
            Some(params) => params,
            None => return None,
        };

        if !self.expect_peek(Token::LBRACE) {
            return None;
        }

        Some(Expression::Func {
            params,
            body: self.parse_block_statement(),
            name: String::from(""),
        })
    }

    fn parse_hash_expr(&mut self) -> Option<Expression> {
        let mut pairs = Vec::new();

        while self.peek_token != Token::RBRACE {
            self.next_token();

            let key = match self.parse_expression(Precedence::Lowest) {
                Some(expr) => expr,
                None => return None,
            };

            if !self.expect_peek(Token::COLON) {
                return None;
            }

            self.next_token();

            let value = match self.parse_expression(Precedence::Lowest) {
                Some(expr) => expr,
                None => return None,
            };

            pairs.push((key, value));

            if self.peek_token != Token::RBRACE && !self.expect_peek(Token::COMMA) {
                return None;
            }
        }

        if !self.expect_peek(Token::RBRACE) {
            return None;
        }

        Some(Expression::Literal(Literal::Hash(pairs)))
    }

    fn parse_array_ident_expr(&mut self) -> Option<Expression> {
        match self.parse_expression_list(Token::RBRACKET) {
            Some(l) => Some(Expression::Literal(Literal::Array(l))),
            None => None,
        }
    }

    fn parse_index_expr(&mut self, left: Expression) -> Option<Expression> {
        self.next_token();

        let index = match self.parse_expression(Precedence::Lowest) {
            Some(expr) => expr,
            None => return None,
        };

        if !self.expect_peek(Token::RBRACKET) {
            return None;
        }

        Some(Expression::Index(Box::new(left), Box::new(index)))
    }

    fn parse_func_params(&mut self) -> Option<Vec<Ident>> {
        let mut params = vec![];
        if self.peek_token == Token::RPAREN {
            self.next_token();
            return Some(params);
        }

        self.next_token();

        match self.parse_ident() {
            Some(ident) => params.push(ident),
            None => return None,
        }

        while self.peek_token == Token::COMMA {
            self.next_token();
            self.next_token();

            match self.parse_ident() {
                Some(ident) => params.push(ident),
                None => return None,
            }
        }

        if !self.expect_peek(Token::RPAREN) {
            return None;
        }

        Some(params)
    }

    fn parse_expression_list(&mut self, end: Token) -> Option<Vec<Expression>> {
        let mut list = vec![];

        if self.peek_token == end {
            self.next_token();
            return Some(list);
        }

        self.next_token();

        match self.parse_expression(Precedence::Lowest) {
            Some(expr) => list.push(expr),
            None => return None,
        }

        while self.peek_token == Token::COMMA {
            self.next_token();
            self.next_token();

            match self.parse_expression(Precedence::Lowest) {
                Some(expr) => list.push(expr),
                None => return None,
            }
        }

        if !self.expect_peek(end) {
            return None;
        }

        Some(list)
    }

    fn parse_block_statement(&mut self) -> BlockStatement {
        self.next_token();
        let mut block = vec![];
        while self.cur_token != Token::RBRACE && self.cur_token != Token::EOF {
            match self.parse_statement() {
                Some(stmt) => block.push(stmt),
                None => {}
            }
            self.next_token();
        }

        block
    }

    fn parse_if_expression(&mut self) -> Option<Expression> {
        if !self.expect_peek(Token::LPAREN) {
            return None;
        }

        self.next_token();

        let cond = match self.parse_expression(Precedence::Lowest) {
            Some(expr) => expr,
            None => return None,
        };

        if !self.expect_peek(Token::RPAREN) || !self.expect_peek(Token::LBRACE) {
            return None;
        }

        let consequence = self.parse_block_statement();
        let mut alternative = None;

        if self.peek_token == Token::ELSE {
            self.next_token();
            if !self.expect_peek(Token::LBRACE) {
                return None;
            }

            alternative = Some(self.parse_block_statement());
        }

        Some(Expression::If {
            condition: Box::new(cond),
            consequence,
            alternative,
        })
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

    fn parse_bool_expr(&mut self) -> Option<Expression> {
        match self.cur_token {
            Token::BOOL(val) => Some(Expression::Literal(Literal::Bool(val))),
            _ => None,
        }
    }

    fn parse_int_expression(&mut self) -> Option<Expression> {
        match self.cur_token {
            Token::INT(i) => Some(Expression::Literal(Literal::Int(i))),
            _ => None,
        }
    }

    fn parse_grouped_expr(&mut self) -> Option<Expression> {
        self.next_token();

        let exp = match self.parse_expression(Precedence::Lowest) {
            Some(expression) => Some(expression),
            _ => None,
        };

        if !self.expect_peek(Token::RPAREN) {
            return None;
        }

        exp
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

    fn parse_string_expression(&mut self) -> Option<Expression> {
        match self.cur_token {
            Token::STRING(ref mut s) => Some(Expression::Literal(Literal::String(s.clone()))),
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
        let a = b;
        "#;

        let l = lexer::Lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 4, "expected 3 statements");
        let expected: [Statement; 4] = [
            Statement::Let(
                Ident(String::from("x")),
                // Expression::Blank,
                Expression::Literal(Literal::Int(5)),
            ),
            Statement::Let(
                Ident(String::from("y")),
                //Expression::Blank,
                Expression::Literal(Literal::Int(10)),
            ),
            Statement::Let(
                Ident(String::from("foobar")),
                //Expression::Blank,
                Expression::Literal(Literal::Int(838383)),
            ),
            Statement::Let(
                Ident(String::from("a")),
                //Expression::Blank,
                Expression::Ident(Ident(String::from("b"))),
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

        let l = lexer::Lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        let errs = p.errors();
        print!("{:?}", errs);
        assert_eq!(errs.len(), 2, "We're expecting two errors");
    }

    #[test]
    fn test_return_statements() {
        let input = r#"
      return 5;
      return 10;
      return 838383;
      return a;
      "#;

        let l = lexer::Lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 4, "expected 3 statements");
        let expected: [Statement; 4] = [
            Statement::Return(
                //Expression::Blank,
                Expression::Literal(Literal::Int(5)),
            ),
            Statement::Return(
                //Expression::Blank,
                Expression::Literal(Literal::Int(10)),
            ),
            Statement::Return(
                //Expression::Blank,
                Expression::Literal(Literal::Int(838383)),
            ),
            Statement::Return(
                //Expression::Blank,
                Expression::Ident(Ident(String::from("a"))),
            ),
        ];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_if_expression() {
        let input = r#"
            if (x < y) {x}
        "#;

        let l = lexer::Lexer::new(String::from(input));
        let mut p = Parser::new(l);

        let program = p.parse_program();

        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 1, "expected 1 statement");

        let expected: [Statement; 1] = [Statement::Expression(Expression::If {
            condition: Box::new(Expression::Infix(
                Infix::Lt,
                Box::new(Expression::Ident(Ident(String::from("x")))),
                Box::new(Expression::Ident(Ident(String::from("y")))),
            )),
            consequence: vec![Statement::Expression(Expression::Ident(Ident(
                String::from("x"),
            )))],
            alternative: None,
        })];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_if_else_expression() {
        let input = r#"
            if (x < y) {x} else {y}
        "#;

        let l = lexer::Lexer::new(String::from(input));
        let mut p = Parser::new(l);

        let program = p.parse_program();

        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 1, "expected 1 statement");

        let expected: [Statement; 1] = [Statement::Expression(Expression::If {
            condition: Box::new(Expression::Infix(
                Infix::Lt,
                Box::new(Expression::Ident(Ident(String::from("x")))),
                Box::new(Expression::Ident(Ident(String::from("y")))),
            )),
            consequence: vec![Statement::Expression(Expression::Ident(Ident(
                String::from("x"),
            )))],
            alternative: Some(vec![Statement::Expression(Expression::Ident(Ident(
                String::from("y"),
            )))]),
        })];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_func_expression() {
        let input = r#"
            fn(x,y) {x + y}
            fn(x) {x}
            fn() {x}
        "#;

        let l = lexer::Lexer::new(String::from(input));
        let mut p = Parser::new(l);

        let program = p.parse_program();

        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 3, "expected 3 statement");

        let expected: [Statement; 3] = [
            Statement::Expression(Expression::Func {
                params: vec![Ident(String::from("x")), Ident(String::from("y"))],
                body: vec![Statement::Expression(Expression::Infix(
                    Infix::Plus,
                    Box::new(Expression::Ident(Ident(String::from("x")))),
                    Box::new(Expression::Ident(Ident(String::from("y")))),
                ))],
                name: String::from(""),
            }),
            Statement::Expression(Expression::Func {
                params: vec![Ident(String::from("x"))],
                body: vec![Statement::Expression(Expression::Ident(Ident(
                    String::from("x"),
                )))],
                name: String::from(""),
            }),
            Statement::Expression(Expression::Func {
                params: vec![],
                body: vec![Statement::Expression(Expression::Ident(Ident(
                    String::from("x"),
                )))],
                name: String::from(""),
            }),
        ];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_call_expression() {
        let input = r#"
            add(1, 2*3, 4+5);
        "#;

        let l = lexer::Lexer::new(String::from(input));
        let mut p = Parser::new(l);

        let program = p.parse_program();

        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 1, "expected 1 statement");

        let expected: [Statement; 1] = [Statement::Expression(Expression::Call {
            func: Box::new(Expression::Ident(Ident(String::from("add")))),
            args: vec![
                Expression::Literal(Literal::Int(1)),
                Expression::Infix(
                    Infix::Multiply,
                    Box::new(Expression::Literal(Literal::Int(2))),
                    Box::new(Expression::Literal(Literal::Int(3))),
                ),
                Expression::Infix(
                    Infix::Plus,
                    Box::new(Expression::Literal(Literal::Int(4))),
                    Box::new(Expression::Literal(Literal::Int(5))),
                ),
            ],
        })];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_array_expression() {
        let input = r#"
            [1, 2 * 2, 3 + 3];
        "#;

        let l = lexer::Lexer::new(String::from(input));
        let mut p = Parser::new(l);

        let program = p.parse_program();

        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 1, "expected 1 statement");

        let expected: [Statement; 1] = [Statement::Expression(Expression::Literal(
            Literal::Array(vec![
                Expression::Literal(Literal::Int(1)),
                Expression::Infix(
                    Infix::Multiply,
                    Box::new(Expression::Literal(Literal::Int(2))),
                    Box::new(Expression::Literal(Literal::Int(2))),
                ),
                Expression::Infix(
                    Infix::Plus,
                    Box::new(Expression::Literal(Literal::Int(3))),
                    Box::new(Expression::Literal(Literal::Int(3))),
                ),
            ]),
        ))];
        for i in 0..program.len() {
            assert_eq!(program[i], expected[i], "{}", i);
        }
    }

    #[test]
    fn test_hash_expression() {
        let input = r#"
            {"abc": "efg", "asdf": 123};
        "#;

        let l = lexer::Lexer::new(String::from(input));
        let mut p = Parser::new(l);

        let program = p.parse_program();

        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 1, "expected 1 statement");

        let expected: [Statement; 1] = [Statement::Expression(Expression::Literal(Literal::Hash(
            vec![
                (
                    Expression::Literal(Literal::String(String::from("abc"))),
                    Expression::Literal(Literal::String(String::from("efg"))),
                ),
                (
                    Expression::Literal(Literal::String(String::from("asdf"))),
                    Expression::Literal(Literal::Int(123)),
                ),
            ],
        )))];
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

        let l = lexer::Lexer::new(String::from(input));

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
      !true;
      !false;
      "#;

        let l = lexer::Lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 4, "expected 2 statements");
        let expected: [Statement; 4] = [
            Statement::Expression(Expression::Prefix(
                Prefix::Minus,
                Box::new(Expression::Ident(Ident(String::from("foobar")))),
            )),
            Statement::Expression(Expression::Prefix(
                Prefix::Bang,
                Box::new(Expression::Literal(Literal::Int(5))),
            )),
            Statement::Expression(Expression::Prefix(
                Prefix::Bang,
                Box::new(Expression::Literal(Literal::Bool(true))),
            )),
            Statement::Expression(Expression::Prefix(
                Prefix::Bang,
                Box::new(Expression::Literal(Literal::Bool(false))),
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
      true == true;
      true != false;
      false == false;
      "#;

        let l = lexer::Lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        assert_eq!(program.len(), 13, "expected 13 statements");
        let expected: [Statement; 13] = [
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
            Statement::Expression(Expression::Infix(
                Infix::Eq,
                Box::new(Expression::Literal(Literal::Bool(true))),
                Box::new(Expression::Literal(Literal::Bool(true))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Ne,
                Box::new(Expression::Literal(Literal::Bool(true))),
                Box::new(Expression::Literal(Literal::Bool(false))),
            )),
            Statement::Expression(Expression::Infix(
                Infix::Eq,
                Box::new(Expression::Literal(Literal::Bool(false))),
                Box::new(Expression::Literal(Literal::Bool(false))),
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
        true;
        false;
        3 > 5 == false;
        3 < 5 == true;
        1 + (2 + 3) + 4;
        (5 + 5) * 2;
        2 / (5 + 5);
        -(5 + 5);
        !(true == true);
      "#;

        let l = lexer::Lexer::new(String::from(input));

        let mut p = Parser::new(l);

        let program = p.parse_program();
        // println!("test test test");
        assert_ne!(program.len(), 0, "No program loaded");
        check_errors(p);

        let expected: [&str; 22] = [
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
            "true",
            "false",
            "((3 > 5) == false)",
            "((3 < 5) == true)",
            "((1 + (2 + 3)) + 4)",
            "((5 + 5) * 2)",
            "(2 / (5 + 5))",
            "(-(5 + 5))",
            "(!(true == true))",
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
