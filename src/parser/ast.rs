use crate::token::Token;

use std::fmt;

#[derive(PartialEq, Clone, Debug)]
pub struct Ident(pub String);

impl From<Token> for Ident {
    fn from(t: Token) -> Self {
        match t {
            Token::IDENT(a) => Self(a),
            _ => Self(String::from("unknown")),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Statement {
    Blank,
    Let(Ident, Expression),
    Return(Expression),
    Expression(Expression),
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Blank => write!(f, ""),
            Self::Let(i, e) => write!(f, "let {} = {};", i.0, e),
            Self::Return(e) => write!(f, "return {};", e),
            Self::Expression(e) => write!(f, "{}", e),
            // _ => write!(f, "D:"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Blank,
    Ident(Ident),
    Literal(Literal),
    Prefix(Prefix, Box<Expression>),
    Infix(Infix, Box<Expression>, Box<Expression>),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Blank => write!(f, ""),
            Self::Ident(i) => write!(f, "{}", i.0),
            Self::Literal(l) => write!(f, "{}", l),
            Self::Prefix(p, e) => {
                write!(f, "({}{})", p, e)
            }
            Self::Infix(i, e1, e2) => {
                write!(f, "({} {} {})", e1, i, e2)
            } // _ => write!(f, "D:"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Prefix {
    Bang,
    Minus,
}

impl fmt::Display for Prefix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Prefix::Minus => write!(f, "-"),
            Prefix::Bang => write!(f, "!"),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Infix {
    Plus,
    Minus,
    Divide,
    Multiply,
    Eq,
    Ne,
    Gte,
    Gt,
    Lte,
    Lt,
}

impl fmt::Display for Infix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Infix::Plus => write!(f, "+"),
            Infix::Minus => write!(f, "-"),
            Infix::Divide => write!(f, "/"),
            Infix::Multiply => write!(f, "*"),
            Infix::Eq => write!(f, "=="),
            Infix::Ne => write!(f, "!="),
            Infix::Gte => write!(f, ">="),
            Infix::Gt => write!(f, ">"),
            Infix::Lte => write!(f, "<="),
            Infix::Lt => write!(f, "<"),
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum Literal {
    Int(i64),
    String(String),
    Bool(bool),
    // Array(Vec<Expression>),
    // Hash(Vec<(Expression, Expression)>),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Int(i) => write!(f, "{}", i),
            Self::String(s) => write!(f, "{}", s),
            Self::Bool(b) => {
                if *b {
                    write!(f, "true")
                } else {
                    write!(f, "false")
                }
            } // _ => write!(f, "D:"),
        }
    }
}

pub type Block_Statement = Vec<Statement>;

pub type Program = Block_Statement;

pub fn precedence_of(t: Token) -> Precedence {
    match t {
        Token::PLUS | Token::MINUS => Precedence::Sum,
        Token::EQ | Token::NE => Precedence::Equals,
        Token::ASTERISK | Token::SLASH => Precedence::Product,
        Token::LT | Token::GT => Precedence::LessGreater,
        Token::LTE | Token::GTE => Precedence::LessGreater,
        // Token::LPAREN
        _ => Precedence::Lowest,
    }
}

#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    Lowest,
    Equals,      // ==
    LessGreater, // > or <
    Sum,         // +
    Product,     // *
    Prefix,      // -X or !X
    Call,        // myFunction(x)
                 // Index,       // array[index]
}
