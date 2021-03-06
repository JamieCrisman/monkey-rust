use crate::code::Instructions;
use crate::evaluator::env::*;
use crate::parser::ast::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub type BuiltInFunc = fn(Vec<Object>) -> Object;

#[derive(PartialEq, Clone, Debug)]
pub enum Object {
    Int(i64),
    String(String),
    Bool(bool),
    Array(Vec<Object>),
    Hash(HashMap<Object, Object>),
    Func(Vec<Ident>, BlockStatement, Rc<RefCell<Env>>, String),
    Builtin(i32, BuiltInFunc),
    Null,
    ReturnValue(Box<Object>),
    Error(String),
    CompiledFunction {
        instructions: Instructions,
        num_locals: i32,
        num_parameters: i32,
    },
    Closure {
        Fn: Box<Object>, // technically specifically ObjectCompiledFunction
        Free: Vec<Object>,
    },
}

impl Object {
    pub fn object_type(&self) -> ObjectType {
        match self {
            Object::Int(_) => ObjectType::Int,
            Object::String(_) => ObjectType::String,
            Object::Bool(_) => ObjectType::Bool,
            Object::Array(_) => ObjectType::Array,
            Object::Hash(_) => ObjectType::Hash,
            Object::Func(_, _, _, _) => ObjectType::Func,
            Object::Builtin(_, _) => ObjectType::Builtin,
            Object::Null => ObjectType::Null,
            Object::ReturnValue(_) => ObjectType::ReturnValue,
            Object::Error(_) => ObjectType::Error,
            Object::CompiledFunction {
                instructions: _,
                num_locals: _,
                num_parameters: _,
            } => ObjectType::CompiledFunction,
            Object::Closure { Fn: _, Free: _ } => ObjectType::Closure,
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
pub enum ObjectType {
    Int,
    String,
    Bool,
    Array,
    Hash,
    Func,
    Builtin,
    Null,
    ReturnValue,
    Error,
    CompiledFunction,
    Closure,
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Object::Int(ref value) => write!(f, "{}", value),
            Object::String(ref value) => write!(f, "{}", value),
            Object::Bool(ref value) => write!(f, "{}", value),
            Object::Array(ref objects) => {
                let mut result = String::new();
                for (i, obj) in objects.iter().enumerate() {
                    if i < 1 {
                        result.push_str(&format!("{}", obj));
                    } else {
                        result.push_str(&format!(", {}", obj));
                    }
                }
                write!(f, "{}", result)
            }
            Object::Hash(ref hash) => {
                let mut result = String::new();
                for (i, (k, v)) in hash.iter().enumerate() {
                    if i < 1 {
                        result.push_str(&format!("{}: {}", k, v));
                    } else {
                        result.push_str(&format!(", {}: {}", k, v));
                    }
                }
                write!(f, "{{{}}}", result)
            }
            Object::Func(ref params, _, _, _) => {
                let mut result = String::new();
                for (i, Ident(ref s)) in params.iter().enumerate() {
                    if i < 1 {
                        result.push_str(&format!("{}", s));
                    } else {
                        result.push_str(&format!(", {}", s));
                    }
                }
                write!(f, "{}({}) {{ ... }}", "func", result)
            }
            Object::Builtin(_, _) => write!(f, "[builtin function]"),
            Object::Null => write!(f, "null"),
            Object::ReturnValue(ref value) => write!(f, "{}", value),
            Object::Error(ref value) => write!(f, "{}", value),
            Object::Closure { Fn: _, Free: _ } => write!(f, "Closure"),
            Object::CompiledFunction {
                instructions: _,
                num_locals,
                num_parameters,
            } => write!(
                f,
                "Compiled Function with {} locals and {} parameters",
                num_locals, num_parameters
            ),
        }
    }
}

impl Eq for Object {}

impl Hash for Object {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            Object::Int(ref i) => i.hash(state),
            Object::Bool(ref b) => b.hash(state),
            Object::String(ref s) => s.hash(state),
            _ => "".hash(state),
        }
    }
}
