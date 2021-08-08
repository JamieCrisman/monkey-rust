use std::fmt;

#[derive(Clone, Debug)]
pub struct Instructions {
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct InstructionList {
    pub data: Vec<Instructions>,
}

impl InstructionList {
    pub fn new() -> Self {
        InstructionList { data: vec![] }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn concat_size(&self) -> usize {
        let mut res: usize = 0;
        for i in self.data.iter() {
            res += i.data.len()
        }
        res
    }

    pub fn push(&mut self, i: Instructions) {
        self.data.push(i)
    }

    pub fn clone(&self) -> Self {
        let mut new_list: Vec<Instructions> = vec![];
        for i in self.data.iter() {
            new_list.push(i.clone());
        }
        InstructionList { data: new_list }
    }
}

impl fmt::Display for InstructionList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.data.len() == 0 {
            return Ok(());
        }
        let mut acc = 0;
        for i in &self.data {
            write!(f, "{:#04} {}\n", acc, i)?;
            acc += i.data.len();
        }
        Ok(())
    }
}

impl Instructions {
    pub fn get_data(&self) -> Vec<u8> {
        self.data.clone()
    }
}

impl fmt::Display for Instructions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.data.len() == 0 {
            return Ok(());
        }
        // let op = Opcode::from_u8(*self.data.get(0).unwrap());

        // println!("converting {}", (*self.data.get(0).unwrap()));
        let op: Opcode = (*self.data.get(0).unwrap()).into();
        let mut values: Vec<u16> = vec![];
        if let Some(ww) = op.width() {
            for w in ww {
                match w {
                    2 => {
                        let data = [*self.data.get(1).unwrap(), *self.data.get(2).unwrap()];
                        values.push(u16::from_be_bytes(data));
                    }
                    _ => {}
                }
            }
        }

        write!(f, "{:?}", op)?;
        for v in values {
            write!(f, " {:?}", v)?;
        }
        Ok(())
    }
}

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Opcode {
    Constant = 0,
    Pop = 1,
    Add = 2,
    Subtract = 3,
    Multiply = 4,
    Divide = 5,
}

impl From<u8> for Opcode {
    fn from(orig: u8) -> Self {
        match orig {
            0 => return Opcode::Constant,
            1 => return Opcode::Pop,
            2 => return Opcode::Add,
            3 => return Opcode::Subtract,
            4 => return Opcode::Multiply,
            5 => return Opcode::Divide,
            _ => panic!("Unknown value: {}", orig),
        };
    }
}

// impl Opcode {
//     fn from_u8(input: &u8) -> Opcode {
//         match input {
//             0u8 => Opcode::Constant,
//             1u8 => Opcode::OpAdd,
//             _ => panic!("Unknown value: {}", input),
//         }
//     }
// }

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum MakeError {
    Reason(String),
}

impl Opcode {
    pub fn width(&self) -> Option<Vec<i16>> {
        match self {
            Opcode::Constant => Some(vec![2]),
            Opcode::Add | Opcode::Divide | Opcode::Subtract | Opcode::Multiply | Opcode::Pop => {
                None
            }
        }
    }
}

// impl fmt::Display for Opcode {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         match *self {
//             Opcode::Constant => write!(f, "{}", Opcode::Constant),
//             Opcode::OpAdd => write!(f, "{}", Opcode::OpAdd),
//         }
//     }
// }

// impl fmt::Display for Instructions {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         // for i in self.len() {}
//     }
// }

pub fn make(op: Opcode, operands: Option<Vec<i32>>) -> Result<Instructions, MakeError> {
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
        if operands.is_none() {
            return Err(MakeError::Reason(format!(
                "Expected operand for opcode {:?}",
                op
            )));
        }

        let mut offset = 1 as usize;
        let ww = op.width().unwrap();
        for (i, o) in operands.unwrap().iter().enumerate() {
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

    Ok(Instructions { data: result })
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    use crate::code::*;
    // use crate::lexer::lexer;
    // use crate::parser::Parser;

    #[test]
    fn test_make() {
        let tests: Vec<(Opcode, Vec<i32>, Instructions, Option<MakeError>)> = vec![
            (
                Opcode::Constant,
                vec![65534],
                Instructions {
                    data: vec![Opcode::Constant as u8, 255, 254],
                },
                None,
            ),
            (
                Opcode::Add,
                vec![],
                Instructions {
                    data: vec![Opcode::Add as u8],
                },
                None,
            ),
        ];

        for test in tests {
            let result: Result<Instructions, MakeError> = make(test.0, Some(test.1));

            if result.is_err() && test.3.is_none() {
                // this will always explode in this case... but I want it to error
                assert_eq!(None, result.as_ref().err());
            }

            let r = result.unwrap();
            assert_eq!(test.2.data.len(), r.data.len());
            println!("{:?}", r.data);
            for (ind, val) in test.2.data.iter().enumerate() {
                assert_eq!(val, r.data.get(ind).expect("expected a value"));
            }
        }
    }

    #[test]
    fn test_display() {
        let instr = InstructionList {
            data: vec![
                make(Opcode::Add, None).unwrap(),
                make(Opcode::Constant, Some(vec![65534])).unwrap(),
                make(Opcode::Constant, Some(vec![1])).unwrap(),
                make(Opcode::Constant, Some(vec![2])).unwrap(),
            ],
        };

        let result = format!("{}", instr);
        assert_eq!(
            "0000 Add\n0001 Constant 65534\n0004 Constant 1\n0007 Constant 2\n",
            result
        );
    }
}
