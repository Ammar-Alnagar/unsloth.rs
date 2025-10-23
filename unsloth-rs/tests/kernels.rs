use ndarray::{Array, IxDyn};
use unsloth_rs::core::Tensor;
use unsloth_rs::kernels::fast_lora::{LoraMlp, LoraQkv};

#[test]
fn test_create_lora_mlp() {
    let _lora_mlp = LoraMlp::new(
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
    );
}

#[test]
fn test_create_lora_qkv() {
    let _lora_qkv = LoraQkv::new(
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
        Tensor::new(Array::zeros(IxDyn(&[0]))),
    );
}
