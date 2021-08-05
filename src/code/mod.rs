use std::fmt;

pub type Instructions = Vec<u8>;

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Opcode {
    Constant,
    OpAdd,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum MakeError {
    Reason(String),
}

impl Opcode {
    pub fn width(&self) -> Option<Vec<i16>> {
        match self {
            Opcode::Constant => Some(vec![2]),
            Opcode::OpAdd => None,
        }
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Opcode::Constant => write!(f, "{}", Opcode::Constant),
            Opcode::OpAdd => write!(f, "{}", Opcode::OpAdd),
        }
    }
}

pub fn make(op: Opcode, operands: Vec<i32>) -> Result<Instructions, MakeError> {
    let mut instruction_len = 1;
    let width = op.width();
    match width {
        Some(size_vec) => {
            for l in size_vec {
                instruction_len += l;
            }
        }
        None => {}
    };
    let mut result = vec![0; instruction_len as usize];
    result[0] = op.clone() as u8;
    if instruction_len > 1 {
        let mut offset = 1 as usize;
        let ww = op.width().unwrap();
        for (i, o) in operands.iter().enumerate() {
            let w = ww.get(i);
            match w {
                Some(v) => {
                    match v {
                        2 => {
                            for (ii, b) in (*o as i16).to_be_bytes().iter().enumerate() {
                                result[offset + ii] = *b;
                            }
                        }
                        _ => {}
                    }
                    offset += *v as usize;
                }
                None => {}
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    use crate::code::*;
    // use crate::lexer::lexer;
    // use crate::parser::Parser;

    #[test]
    fn test_make() {
        let tests: Vec<(Opcode, Vec<i32>, Vec<u8>, Option<MakeError>)> = vec![(
            Opcode::Constant,
            vec![65534],
            vec![Opcode::Constant as u8, 255, 254],
            None,
        )];

        for test in tests {
            let result: Result<Vec<u8>, MakeError> = make(test.0, test.1);

            if result.is_err() && test.3.is_none() {
                // this will always explode in this case... but I want it to error
                assert_eq!(None, result.as_ref().err());
            }

            let r = result.unwrap();
            assert_eq!(test.2.len(), r.len());
            println!("{:?}", r);
            for (ind, val) in test.2.iter().enumerate() {
                assert_eq!(val, r.get(ind).expect("expected a value"));
            }
        }
    }
}
