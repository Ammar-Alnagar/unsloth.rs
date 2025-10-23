use unsloth_rs::models::llama::LlamaModel;
use unsloth_rs::trainer::Trainer;

#[test]
fn test_create_trainer() {
    let llama_model = LlamaModel::new();
    let trainer = Trainer::new(llama_model);
    trainer.train();
}
