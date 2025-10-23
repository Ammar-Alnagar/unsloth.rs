use crate::models::llama::LlamaModel;
use crate::core::Tensor;

pub struct Trainer {
    model: LlamaModel,
}

impl Trainer {
    pub fn new(model: LlamaModel) -> Self {
        Trainer { model }
    }

    pub fn train(&self) {
        // We will implement this later.
        println!("Training the model...");
    }
}
