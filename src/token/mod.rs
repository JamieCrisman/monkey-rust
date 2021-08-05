#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Token {
    ILLEGAL,
    EOF,

    IDENT(String),
    INT(i64),
    STRING(String),
    BOOL(bool),

    ASSIGN,
    PLUS,
    MINUS,
    BANG,
    ASTERISK,
    SLASH,

    LT,
    LTE,
    GT,
    GTE,

    COMMA,
    COLON,
    SEMICOLON,

    LPAREN,
    RPAREN,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,

    FUNCTION,
    LET,
    IF,
    ELSE,
    RETURN,
    EQ,
    NE,
}
