use clap::Parser;
use ndarray::{Array, IxDyn};
use unsloth_rs::core::Tensor;
use unsloth_rs::dataprep::synthetic::SyntheticDataKit;
use unsloth_rs::kernels::fast_lora::{LoraMlp, LoraQkv};
use unsloth_rs::models::llama::LlamaModel;
use unsloth_rs::rl::ppo::PPO;
use unsloth_rs::save::Model;
use unsloth_rs::trainer::Trainer;
use unsloth_rs::utils::hf_hub;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: String,
}

fn main() {
    let args = Args::parse();
    println!("Model: {}", args.model);

    let tensor = Tensor::new(
        Array::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );
    println!("Tensor: {:?}", tensor);

    let synthetic_data_kit = SyntheticDataKit::new();
    synthetic_data_kit.prepare_qa_generation();
    println!("Synthetic data generation complete!");

    let lora_mlp = LoraMlp::new(
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
    );
    let lora_qkv = LoraQkv::new(
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
        Tensor::new(Array::zeros(IxDyn(&[2, 2]))),
    );

    let mlp_output = lora_mlp.forward(&tensor);
    let (q, k, v) = lora_qkv.forward(&tensor);

    println!("MLP output: {:?}", mlp_output);
    println!("Q: {:?}", q);
    println!("K: {:?}", k);
    println!("V: {:?}", v);

    let model = Model::new();
    model.save("model.bin");
    let loaded_model = Model::load("model.bin");

    let llama_model = LlamaModel::new(8, 4, 128, 32000, 32);
    let llama_output = llama_model.forward(&[1, 2, 3, 4]);
    println!("Llama output: {:?}", llama_output);

    let ppo = PPO::new();
    ppo.train();

    let trainer = Trainer::new(llama_model);
    trainer.train();

    hf_hub::get_model_info("unsloth/llama-3-8b-bnb-4bit");
    hf_hub::list_models();
}
