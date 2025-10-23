use unsloth_rs::models::llama::LlamaModel;

use unsloth_rs::core::Tensor;
use unsloth_rs::models::llama::LlamaAttention;
use ndarray::{Array, IxDyn};

#[test]
fn test_create_llama_model() {
    let _llama_model = unsloth_rs::models::llama::LlamaModel::new(8, 4, 128, 32000, 32);
}

#[test]
fn test_llama_attention() {
    let n_heads = 8;
    let n_kv_heads = 4;
    let head_dim = 128;
    let seq_len = 10;
    let hidden_dim = n_heads * head_dim;

    let mut attention = LlamaAttention::new(n_heads, n_kv_heads, head_dim);
    attention.wq = Tensor::new(Array::zeros(IxDyn(&[hidden_dim, n_heads * head_dim])));
    attention.wk = Tensor::new(Array::zeros(IxDyn(&[hidden_dim, n_kv_heads * head_dim])));
    attention.wv = Tensor::new(Array::zeros(IxDyn(&[hidden_dim, n_kv_heads * head_dim])));
    attention.wo = Tensor::new(Array::zeros(IxDyn(&[n_heads * head_dim, hidden_dim])));

    let input = Tensor::new(Array::zeros(IxDyn(&[seq_len, hidden_dim])));
    let output = attention.forward(&input);

    assert_eq!(output.data.shape(), &[seq_len, hidden_dim]);
}
