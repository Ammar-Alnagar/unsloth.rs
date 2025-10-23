use crate::core::Tensor;

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
        // We will implement this later.
        Tensor::new(vec![], vec![])
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
        // We will implement this later.
        (
            Tensor::new(vec![], vec![]),
            Tensor::new(vec![], vec![]),
            Tensor::new(vec![], vec![]),
        )
    }
}
