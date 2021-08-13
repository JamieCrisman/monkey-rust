use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolScope {
    Global,
    Local,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub scope: SymbolScope,
    pub index: usize,
}

#[derive(Clone, Debug)]
pub struct SymbolTable {
    store: HashMap<String, Symbol>,
    pub num_definitions: usize,
    pub outer: Option<Rc<RefCell<SymbolTable>>>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            num_definitions: 0,
            outer: None,
        }
    }

    pub fn new_with_outer(outer: Rc<RefCell<SymbolTable>>) -> Self {
        Self {
            store: HashMap::new(),
            num_definitions: 0,
            outer: Some(outer),
        }
    }

    pub fn define(&mut self, name: &str) -> Symbol {
        let scope = match self.outer {
            None => SymbolScope::Global,
            Some(_) => SymbolScope::Local,
        };

        let result = Symbol {
            name: String::from(name),
            index: self.num_definitions,
            scope,
        };
        self.store.insert(result.name.clone(), result.clone());
        self.num_definitions += 1;
        result
    }

    pub fn resolve(&self, name: String) -> Option<Symbol> {
        match self.store.get(&name) {
            Some(value) => Some(value.clone()),
            None => match self.outer {
                Some(ref outer) => outer.borrow().resolve(name),
                None => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    // use crate::evaluator::builtins::new_builtins;
    use crate::compiler::symbol_table::*;

    #[test]
    fn test_define() {
        let mut expected: HashMap<String, Symbol> = HashMap::new();
        expected.insert(
            "a".to_string(),
            Symbol {
                index: 0,
                name: String::from("a"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "b".to_string(),
            Symbol {
                index: 1,
                name: String::from("b"),
                scope: SymbolScope::Global,
            },
        );

        let mut global = SymbolTable::new();
        let a = global.define("a");
        assert_eq!(a, expected["a"]);
        let b = global.define("b");
        assert_eq!(b, expected["b"]);
    }

    #[test]
    fn test_resolve() {
        let mut expected: HashMap<String, Symbol> = HashMap::new();
        expected.insert(
            "a".to_string(),
            Symbol {
                index: 0,
                name: String::from("a"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "b".to_string(),
            Symbol {
                index: 1,
                name: String::from("b"),
                scope: SymbolScope::Global,
            },
        );

        let mut global = SymbolTable::new();
        global.define("a");
        global.define("b");

        for (k, v) in expected.iter() {
            assert_eq!(
                (global
                    .resolve(k.to_string())
                    .expect("expected to get a value")),
                *v
            );
        }
    }

    #[test]
    fn test_local_define() {
        let mut expected: HashMap<String, Symbol> = HashMap::new();
        expected.insert(
            "a".to_string(),
            Symbol {
                index: 0,
                name: String::from("a"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "b".to_string(),
            Symbol {
                index: 1,
                name: String::from("b"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "c".to_string(),
            Symbol {
                index: 0,
                name: String::from("c"),
                scope: SymbolScope::Local,
            },
        );
        expected.insert(
            "d".to_string(),
            Symbol {
                index: 1,
                name: String::from("d"),
                scope: SymbolScope::Local,
            },
        );

        let mut global = SymbolTable::new();
        let a = global.define("a");
        assert_eq!(a, expected["a"]);
        let b = global.define("b");
        assert_eq!(b, expected["b"]);

        let mut local = SymbolTable::new_with_outer(Rc::new(RefCell::new(global)));

        // let a2 = local.define("a");
        // assert_eq!(a2, expected["a"]);
        // let b2 = local.define("b");
        // assert_eq!(b2, expected["b"]);

        let c = local.define("c");
        assert_eq!(c, expected["c"]);
        let d = local.define("d");
        assert_eq!(d, expected["d"]);
        assert_eq!(local.resolve(String::from("a")).unwrap(), expected["a"]);
        assert_eq!(local.resolve(String::from("d")).unwrap(), expected["d"]);
    }

    #[test]
    fn test_sublocal_define() {
        let mut expected: HashMap<String, Symbol> = HashMap::new();
        expected.insert(
            "a".to_string(),
            Symbol {
                index: 0,
                name: String::from("a"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "b".to_string(),
            Symbol {
                index: 1,
                name: String::from("b"),
                scope: SymbolScope::Global,
            },
        );
        expected.insert(
            "c".to_string(),
            Symbol {
                index: 0,
                name: String::from("c"),
                scope: SymbolScope::Local,
            },
        );
        expected.insert(
            "d".to_string(),
            Symbol {
                index: 1,
                name: String::from("d"),
                scope: SymbolScope::Local,
            },
        );
        expected.insert(
            "e".to_string(),
            Symbol {
                index: 0,
                name: String::from("e"),
                scope: SymbolScope::Local,
            },
        );
        expected.insert(
            "f".to_string(),
            Symbol {
                index: 1,
                name: String::from("f"),
                scope: SymbolScope::Local,
            },
        );

        let mut global = SymbolTable::new();
        global.define("a");
        global.define("b");
        // assert_eq!(a, expected["a"]);
        // assert_eq!(b, expected["b"]);

        let mut local = SymbolTable::new_with_outer(Rc::new(RefCell::new(global)));
        local.define("c");
        local.define("d");

        let mut local2 = SymbolTable::new_with_outer(Rc::new(RefCell::new(local)));
        local2.define("e");
        local2.define("f");
        // let a2 = local.define("a");
        // assert_eq!(a2, expected["a"]);
        // let b2 = local.define("b");
        // assert_eq!(b2, expected["b"]);

        // assert_eq!(c, expected["c"]);
        // assert_eq!(d, expected["d"]);
        assert_eq!(local2.resolve(String::from("a")).unwrap(), expected["a"]);
        assert_eq!(local2.resolve(String::from("b")).unwrap(), expected["b"]);
        assert_eq!(local2.resolve(String::from("c")).unwrap(), expected["c"]);
        assert_eq!(local2.resolve(String::from("d")).unwrap(), expected["d"]);
        assert_eq!(local2.resolve(String::from("e")).unwrap(), expected["e"]);
        assert_eq!(local2.resolve(String::from("f")).unwrap(), expected["f"]);
    }
}
