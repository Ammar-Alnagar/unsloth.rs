use unsloth_rs::models::llama::LlamaModel;
use unsloth_rs::trainer::Trainer;

#[test]
fn test_create_trainer() {
    let llama_model = LlamaModel::new(8, 4, 128, 32000, 32);
    let trainer = Trainer::new(llama_model);
    trainer.train();
}
