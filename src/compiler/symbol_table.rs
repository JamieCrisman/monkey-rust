use std::collections::HashMap;

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
    num_definitions: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            num_definitions: 0,
        }
    }

    pub fn define(&mut self, name: &str) -> Symbol {
        let result = Symbol {
            name: String::from(name),
            index: self.num_definitions,
            scope: SymbolScope::Global,
        };
        self.store.insert(result.name.clone(), result.clone());
        self.num_definitions += 1;
        result
    }

    pub fn resolve(&mut self, name: &str) -> Option<&Symbol> {
        self.store.get(&String::from(name))
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
            assert_eq!(*(global.resolve(k).expect("expected to get a value")), *v);
        }
    }
}
