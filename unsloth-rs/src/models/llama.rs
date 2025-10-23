use crate::core::Tensor;
use ndarray::{s, Array, Array3, Ix3, IxDyn};

pub struct LlamaAttention {
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

impl LlamaAttention {
    pub fn new(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let rotary_dim = head_dim; // Typically the same as head_dim
        let wq = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let wk = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let wv = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let wo = Tensor::new(Array::zeros(IxDyn(&[1, 1])));

        LlamaAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            rotary_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (seq_len, hidden_dim) = (x.data.shape()[0], x.data.shape()[1]);
        let x_reshaped =
            Tensor::new(x.data.clone().into_shape((seq_len, hidden_dim)).unwrap().into_dyn());

        let mut q_proj = x_reshaped.matmul(&self.wq);
        let mut k_proj = x_reshaped.matmul(&self.wk);
        let v_proj = x_reshaped.matmul(&self.wv);

        q_proj = q_proj.rope(0, self.rotary_dim, 2048, 10000.0);
        k_proj = k_proj.rope(0, self.rotary_dim, 2048, 10000.0);

        // Reshape: [seq_len, n_heads, head_dim]
        let q = q_proj.data.into_shape((seq_len, self.n_heads, self.head_dim)).unwrap();
        let k = k_proj.data.into_shape((seq_len, self.n_kv_heads, self.head_dim)).unwrap();
        let v = v_proj.data.into_shape((seq_len, self.n_kv_heads, self.head_dim)).unwrap();

        // Transpose to: [n_heads, seq_len, head_dim]
        let q = q.permuted_axes([1, 0, 2]);
        let k = k.permuted_axes([1, 0, 2]);
        let v = v.permuted_axes([1, 0, 2]);

        // Repeat K and V to match number of Q heads
        let n_rep = self.n_heads / self.n_kv_heads;

        // CRITICAL: Make K and V contiguous before reshaping
        let k = k.as_standard_layout().to_owned();
        let v = v.as_standard_layout().to_owned();

        // Reshape K and V: [n_kv_heads, seq_len, head_dim] -> [n_kv_heads, 1, seq_len, head_dim]
        let k_reshaped = k.into_shape((self.n_kv_heads, 1, seq_len, self.head_dim)).unwrap();
        let v_reshaped = v.into_shape((self.n_kv_heads, 1, seq_len, self.head_dim)).unwrap();

        // Broadcast: [n_kv_heads, 1, seq_len, head_dim] -> [n_kv_heads, n_rep, seq_len, head_dim]
        let k_broadcasted =
            k_reshaped.broadcast((self.n_kv_heads, n_rep, seq_len, self.head_dim)).unwrap();
        let v_broadcasted =
            v_reshaped.broadcast((self.n_kv_heads, n_rep, seq_len, self.head_dim)).unwrap();

        // Materialize the broadcast before reshaping
        // Flatten: [n_kv_heads, n_rep, seq_len, head_dim] -> [n_heads, seq_len, head_dim]
        let k_repeated = k_broadcasted
            .to_owned()
            .into_shape((self.n_heads, seq_len, self.head_dim))
            .unwrap();
        let v_repeated = v_broadcasted
            .to_owned()
            .into_shape((self.n_heads, seq_len, self.head_dim))
            .unwrap();

        // Compute attention scores: Q @ K^T
        // Q: [n_heads, seq_len, head_dim]
        // K^T: [n_heads, head_dim, seq_len]
        // Result: [n_heads, seq_len, seq_len]

        let k_t = k_repeated.permuted_axes([0, 2, 1]); // [n_heads, head_dim, seq_len]

        // Batch matmul for each head
        let mut scores = Array3::<f32>::zeros((self.n_heads, seq_len, seq_len));
        for h in 0..self.n_heads {
            let q_h = q.slice(s![h, .., ..]);
            let k_h = k_t.slice(s![h, .., ..]);
            scores.slice_mut(s![h, .., ..]).assign(&q_h.dot(&k_h));
        }

        scores = scores / (self.head_dim as f32).sqrt();

        // Apply softmax over last dimension
        let scores_tensor = Tensor::new(scores.into_dyn());
        let attention_weights = scores_tensor.softmax(2);

        // Attention weights @ V
        // attention_weights: [n_heads, seq_len, seq_len]
        // V: [n_heads, seq_len, head_dim]
        // Result: [n_heads, seq_len, head_dim]

        let attn_3d = attention_weights.data.into_dimensionality::<Ix3>().unwrap();
        let mut attention_output = Array3::<f32>::zeros((self.n_heads, seq_len, self.head_dim));

        for h in 0..self.n_heads {
            let attn_h = attn_3d.slice(s![h, .., ..]);
            let v_h = v_repeated.slice(s![h, .., ..]);
            attention_output.slice_mut(s![h, .., ..]).assign(&attn_h.dot(&v_h));
        }

        // Transpose back: [n_heads, seq_len, head_dim] -> [seq_len, n_heads, head_dim]
        let attention_output = attention_output.permuted_axes([1, 0, 2]);

        // Flatten: [seq_len, n_heads, head_dim] -> [seq_len, n_heads * head_dim]
        let attention_output = attention_output
            .as_standard_layout()
            .to_owned()
            .into_shape((seq_len, self.n_heads * self.head_dim))
            .unwrap();

        let final_output = Tensor::new(attention_output.into_dyn()).matmul(&self.wo);
        final_output
    }
}

pub struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    attention_norm: Tensor,
    ffn_norm: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
}

impl LlamaDecoderLayer {
    pub fn new(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let attention_norm = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let ffn_norm = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let w1 = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let w2 = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let w3 = Tensor::new(Array::zeros(IxDyn(&[1, 1])));

        LlamaDecoderLayer {
            self_attn: LlamaAttention::new(n_heads, n_kv_heads, head_dim),
            attention_norm,
            ffn_norm,
            w1,
            w2,
            w3,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = x.rmsnorm(&self.attention_norm, 1e-5);
        let attention_output = self.self_attn.forward(&h);
        let h = x.add(&attention_output);

        let h_norm = h.rmsnorm(&self.ffn_norm, 1e-5);
        let gate = h_norm.matmul(&self.w1).silu();
        let up = h_norm.matmul(&self.w3);
        let ff = gate.mul(&up);
        let ff = ff.matmul(&self.w2);

        h.add(&ff)
    }
}

pub struct LlamaModel {
    embedding: Tensor,
    layers: Vec<LlamaDecoderLayer>,
    norm: Tensor,
    output: Tensor,
}

impl LlamaModel {
    pub fn new(n_heads: usize, n_kv_heads: usize, head_dim: usize, vocab_size: usize, n_layers: usize) -> Self {
        let embedding = Tensor::new(Array::zeros(IxDyn(&[vocab_size, n_heads * head_dim])));
        let layers = (0..n_layers).map(|_| LlamaDecoderLayer::new(n_heads, n_kv_heads, head_dim)).collect();
        let norm = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        let output = Tensor::new(Array::zeros(IxDyn(&[1, 1])));
        LlamaModel {
            embedding,
            layers,
            norm,
            output,
        }
    }

    pub fn forward(&self, x: &[usize]) -> Tensor {
        let mut h = self.embedding.data.select(ndarray::Axis(0), x);
        let mut h = Tensor::new(h.to_owned().into_dyn());
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h = h.rmsnorm(&self.norm, 1e-5);
        h.matmul(&self.output)
    }
}
