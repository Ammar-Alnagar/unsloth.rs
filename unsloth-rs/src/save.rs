use crate::core::Tensor;
use std::collections::HashMap;

pub struct Model {
    tensors: HashMap<String, Tensor>,
}

impl Model {
    pub fn new() -> Self {
        Model {
            tensors: HashMap::new(),
        }
    }

    pub fn save(&self, filepath: &str) {
        // We will implement this later.
        println!("Saving model to {}", filepath);
    }

    pub fn load(filepath: &str) -> Self {
        // We will implement this later.
        println!("Loading model from {}", filepath);
        Model {
            tensors: HashMap::new(),
        }
    }
}
