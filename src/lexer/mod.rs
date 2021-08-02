use crate::token::Token;
pub struct lexer {
    input: String,
    position: usize,
    read_pos: usize,
    ch: Option<char>,
}

fn is_letter(ch: char) -> bool {
    'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_'
}

fn is_digit(ch: char) -> bool {
    '0' <= ch && ch <= '9'
}

impl lexer {
    pub fn new(input: String) -> Self {
        let mut s = Self {
            input,
            position: 0,
            read_pos: 0,
            ch: None,
        };
        s.read_char();
        s
    }

    fn read_char(&mut self) {
        if self.read_pos >= self.input.len() {
            self.ch = None;
        } else {
            self.ch = self.input.chars().nth(self.read_pos);
        }
        self.position = self.read_pos;
        self.read_pos += 1;
    }

    fn skip_whitespace(&mut self) {
        while self.ch.is_some() {
            match self.ch.unwrap() {
                ' ' | '\t' | '\n' | '\r' => {
                    // println!("clearing whitespace");
                    self.read_char();
                }
                _ => {
                    break;
                }
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        if self.read_pos >= self.input.len() {
            None
        } else {
            self.input.chars().nth(self.read_pos)
        }
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();
        let mut maintainCh = false;
        let t = if self.ch.is_none() {
            Token::EOF
        } else {
            match self.ch.unwrap() {
                '=' => {
                    let peeked = self.peek_char();
                    if peeked.is_some() && peeked.unwrap() == '=' {
                        self.read_char();
                        Token::EQ
                    } else {
                        Token::ASSIGN
                    }
                }
                ';' => Token::SEMICOLON,
                '(' => Token::LPAREN,
                ')' => Token::RPAREN,
                '{' => Token::LBRACE,
                '}' => Token::RBRACE,
                '[' => Token::LBRACKET,
                ']' => Token::RBRACKET,
                '+' => Token::PLUS,
                '-' => Token::MINUS,
                '/' => Token::SLASH,
                '*' => Token::ASTERISK,
                '!' => {
                    let peeked = self.peek_char();
                    if peeked.is_some() && peeked.unwrap() == '=' {
                        self.read_char();
                        Token::NE
                    } else {
                        Token::BANG
                    }
                }
                '>' => {
                    let peeked = self.peek_char();
                    if peeked.is_some() && peeked.unwrap() == '=' {
                        self.read_char();
                        Token::GTE
                    } else {
                        Token::GT
                    }
                }
                '<' => {
                    let peeked = self.peek_char();
                    if peeked.is_some() && peeked.unwrap() == '=' {
                        self.read_char();
                        Token::LTE
                    } else {
                        Token::LT
                    }
                }
                ',' => Token::COMMA,
                '"' => {
                    return self.read_string();
                }
                x => {
                    maintainCh = true;
                    if is_letter(x) {
                        let literal = self.read_identifier();
                        match literal {
                            "fn" => Token::FUNCTION,
                            "let" => Token::LET,
                            "true" => Token::BOOL(true),
                            "false" => Token::BOOL(false),
                            "if" => Token::IF,
                            "else" => Token::ELSE,
                            "return" => Token::RETURN,
                            _ => Token::IDENT(String::from(literal)),
                        }
                    } else if is_digit(x) {
                        let literal = self.read_number();
                        Token::INT(literal.parse::<i64>().unwrap())
                    } else {
                        Token::ILLEGAL
                    }
                }
            }
        };
        if !maintainCh {
            self.read_char();
        }

        t
    }

    fn read_string(&mut self) -> Token {
        self.read_char();

        let spos = self.position;
        loop {
            match self.ch {
                Some('"') | None => {
                    // TODO: hmm... not good to clone the entire input then grab the specific bit
                    let literal = &self.input.clone()[spos..self.position];
                    self.read_char();
                    return Token::STRING(literal.to_string());
                }
                _ => {
                    self.read_char();
                }
            }
        }
    }

    fn read_identifier(&mut self) -> &str {
        let pos = self.position;
        while self.ch.is_some() && is_letter(self.ch.unwrap()) {
            self.read_char();
        }
        return &self.input[pos..self.position];
    }

    fn read_number(&mut self) -> &str {
        let pos = self.position;
        while self.ch.is_some() && is_digit(self.ch.unwrap()) {
            self.read_char();
        }
        return &self.input[pos..self.position];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_next_token_basics() {
        let input = "=+(){},;";
        let expected: [Token; 9] = [
            Token::ASSIGN,
            Token::PLUS,
            Token::LPAREN,
            Token::RPAREN,
            Token::LBRACE,
            Token::RBRACE,
            Token::COMMA,
            Token::SEMICOLON,
            Token::EOF,
        ];

        let mut r = lexer::new(String::from(input));
        for i in 0..expected.len() {
            let next = r.next_token();
            assert_eq!(expected[i], next, "{}", i);
        }
    }

    #[test]
    fn test_next_token_more_symbols() {
        let input = "+-><!/*==!=>=<=";
        let expected: [Token; 12] = [
            Token::PLUS,
            Token::MINUS,
            Token::GT,
            Token::LT,
            Token::BANG,
            Token::SLASH,
            Token::ASTERISK,
            Token::EQ,
            Token::NE,
            Token::GTE,
            Token::LTE,
            Token::EOF,
        ];

        let mut r = lexer::new(String::from(input));
        for i in 0..expected.len() {
            let next = r.next_token();
            assert_eq!(expected[i], next, "{}", i);
        }
    }

    #[test]
    fn test_next_token_code() {
        let input = r#"let five = 5;
      let ten = 10;

      let add = fn(x, y) {
        x + y;
      };
      
      let result = add(five, ten); 
      if (a == b) {
        5;
      }
      "test";
      [1, 2];
      abc[1];
      def["asdf"];
      "#;

        let expected = vec![
            Token::LET,
            Token::IDENT(String::from("five")),
            Token::ASSIGN,
            Token::INT(5),
            Token::SEMICOLON,
            Token::LET,
            Token::IDENT(String::from("ten")),
            Token::ASSIGN,
            Token::INT(10),
            Token::SEMICOLON,
            Token::LET,
            Token::IDENT(String::from("add")),
            Token::ASSIGN,
            Token::FUNCTION,
            Token::LPAREN,
            Token::IDENT(String::from("x")),
            Token::COMMA,
            Token::IDENT(String::from("y")),
            Token::RPAREN,
            Token::LBRACE,
            Token::IDENT(String::from("x")),
            Token::PLUS,
            Token::IDENT(String::from("y")),
            Token::SEMICOLON,
            Token::RBRACE,
            Token::SEMICOLON,
            Token::LET,
            Token::IDENT(String::from("result")),
            Token::ASSIGN,
            Token::IDENT(String::from("add")),
            Token::LPAREN,
            Token::IDENT(String::from("five")),
            Token::COMMA,
            Token::IDENT(String::from("ten")),
            Token::RPAREN,
            Token::SEMICOLON,
            Token::IF,
            Token::LPAREN,
            Token::IDENT(String::from("a")),
            Token::EQ,
            Token::IDENT(String::from("b")),
            Token::RPAREN,
            Token::LBRACE,
            Token::INT(5),
            Token::SEMICOLON,
            Token::RBRACE,
            Token::STRING(String::from("test")),
            Token::SEMICOLON,
            Token::LBRACKET,
            Token::INT(1),
            Token::COMMA,
            Token::INT(2),
            Token::RBRACKET,
            Token::SEMICOLON,
            Token::IDENT(String::from("abc")),
            Token::LBRACKET,
            Token::INT(1),
            Token::RBRACKET,
            Token::SEMICOLON,
            Token::IDENT(String::from("def")),
            Token::LBRACKET,
            Token::STRING(String::from("asdf")),
            Token::RBRACKET,
            Token::SEMICOLON,
            Token::EOF,
        ];

        let mut r = lexer::new(String::from(input));
        for i in 0..expected.len() {
            let next = r.next_token();
            // println!("{:?}", next);
            assert_eq!(expected[i], next, "{}", i);
        }
    }
}
