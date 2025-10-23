use crate::core::Tensor;
use ndarray::{Array, IxDyn};

pub struct LoraMlp {
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    lora_a_gate: Tensor,
    lora_b_gate: Tensor,
    lora_a_up: Tensor,
    lora_b_up: Tensor,
    lora_a_down: Tensor,
    lora_b_down: Tensor,
}

impl LoraMlp {
    pub fn new(
        gate_w: Tensor,
        up_w: Tensor,
        down_w: Tensor,
        lora_a_gate: Tensor,
        lora_b_gate: Tensor,
        lora_a_up: Tensor,
        lora_b_up: Tensor,
        lora_a_down: Tensor,
        lora_b_down: Tensor,
    ) -> Self {
        LoraMlp {
            gate_w,
            up_w,
            down_w,
            lora_a_gate,
            lora_b_gate,
            lora_a_up,
            lora_b_up,
            lora_a_down,
            lora_b_down,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Gate path
        let gate_main = x.matmul(&self.gate_w);
        let gate_lora = x.matmul(&self.lora_a_gate).matmul(&self.lora_b_gate);
        let gate = gate_main.add(&gate_lora);

        // Up path
        let up_main = x.matmul(&self.up_w);
        let up_lora = x.matmul(&self.lora_a_up).matmul(&self.lora_b_up);
        let up = up_main.add(&up_lora);

        // Activation
        let act = gate.silu().mul(&up);

        // Down path
        let down_main = act.matmul(&self.down_w);
        let down_lora = act.matmul(&self.lora_a_down).matmul(&self.lora_b_down);
        down_main.add(&down_lora)
    }
}

pub struct LoraQkv {
    q_w: Tensor,
    k_w: Tensor,
    v_w: Tensor,
    lora_a_q: Tensor,
    lora_b_q: Tensor,
    lora_a_k: Tensor,
    lora_b_k: Tensor,
    lora_a_v: Tensor,
    lora_b_v: Tensor,
}

impl LoraQkv {
    pub fn new(
        q_w: Tensor,
        k_w: Tensor,
        v_w: Tensor,
        lora_a_q: Tensor,
        lora_b_q: Tensor,
        lora_a_k: Tensor,
        lora_b_k: Tensor,
        lora_a_v: Tensor,
        lora_b_v: Tensor,
    ) -> Self {
        LoraQkv {
            q_w,
            k_w,
            v_w,
            lora_a_q,
            lora_b_q,
            lora_a_k,
            lora_b_k,
            lora_a_v,
            lora_b_v,
        }
    }

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        // Q path
        let q_main = x.matmul(&self.q_w);
        let q_lora = x.matmul(&self.lora_a_q).matmul(&self.lora_b_q);
        let q = q_main.add(&q_lora);

        // K path
        let k_main = x.matmul(&self.k_w);
        let k_lora = x.matmul(&self.lora_a_k).matmul(&self.lora_b_k);
        let k = k_main.add(&k_lora);

        // V path
        let v_main = x.matmul(&self.v_w);
        let v_lora = x.matmul(&self.lora_a_v).matmul(&self.lora_b_v);
        let v = v_main.add(&v_lora);

        (q, k, v)
    }
}
