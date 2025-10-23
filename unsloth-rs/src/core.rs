use ndarray::{Array, Ix2, IxDyn};

#[derive(Debug)]
pub struct Tensor {
    pub data: Array<f32, IxDyn>,
}

impl Tensor {
    pub fn new(data: Array<f32, IxDyn>) -> Self {
        Tensor { data }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();

        // Ensure the arrays are 2D for matmul
        assert_eq!(self_shape.len(), 2, "Matmul inputs must be 2D");
        assert_eq!(other_shape.len(), 2, "Matmul inputs must be 2D");
        assert_eq!(
            self_shape[1], other_shape[0],
            "Matmul dimensions are incompatible"
        );

        let self_2d = self
            .data
            .view()
            .into_shape((self_shape[0], self_shape[1]))
            .unwrap();
        let other_2d = other
            .data
            .view()
            .into_shape((other_shape[0], other_shape[1]))
            .unwrap();

        let result = self_2d.dot(&other_2d);
        Tensor::new(result.into_dyn())
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let result = &self.data + &other.data;
        Tensor::new(result)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let result = &self.data * &other.data;
        Tensor::new(result)
    }

    pub fn silu(&self) -> Tensor {
        let result = self.data.mapv(|x| x / (1.0 + (-x).exp()));
        Tensor::new(result)
    }

    pub fn rmsnorm(&self, weight: &Tensor, epsilon: f32) -> Tensor {
        let last_dim = self.data.ndim() - 1;
        let variance = self.data.mapv(|x| x.powi(2)).mean_axis(ndarray::Axis(last_dim)).unwrap();
        let rrms = (variance + epsilon).mapv(f32::sqrt).mapv(|x| 1.0 / x);
        let rrms_reshaped = rrms.insert_axis(ndarray::Axis(last_dim));
        let normalized_x = &self.data * &rrms_reshaped;

        let weight_shape = weight.data.shape();
        let reshaped_weight = weight.data.view().into_shape((1, weight_shape[0])).unwrap();

        let result = &normalized_x * &reshaped_weight;
        Tensor::new(result.into_dyn())
    }

    pub fn rope(&self, pos: usize, rotary_dim: usize, max_seq_len: usize, theta: f32) -> Tensor {
        let mut new_data = self.data.clone();
        let inv_freq: Vec<f32> = (0..rotary_dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();

        let freqs: Vec<f32> = inv_freq.iter().map(|&inv_freq_val| (pos as f32) * inv_freq_val).collect();
        let cos_vals: Vec<f32> = freqs.iter().map(|f| f.cos()).collect();
        let sin_vals: Vec<f32> = freqs.iter().map(|f| f.sin()).collect();

        for mut seq_slice in new_data.outer_iter_mut() {
            for i in 0..(rotary_dim / 2) {
                let cos = cos_vals[i];
                let sin = sin_vals[i];

                let x1 = seq_slice[i * 2];
                let x2 = seq_slice[i * 2 + 1];

                seq_slice[i * 2] = x1 * cos - x2 * sin;
                seq_slice[i * 2 + 1] = x2 * cos + x1 * sin;
            }
        }

        Tensor::new(new_data)
    }

    pub fn softmax(&self, axis: usize) -> Tensor {
        let mut new_data = self.data.clone();

        // Special case for 1D arrays - apply softmax to entire array
        if new_data.ndim() == 1 {
            let max_val = new_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            new_data.mapv_inplace(|x| (x - max_val).exp());
            let sum = new_data.sum();
            new_data.mapv_inplace(|x| x / sum);
        } else {
            new_data.axis_iter_mut(ndarray::Axis(axis)).for_each(|mut lane| {
                let max_val = lane.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                lane.mapv_inplace(|x| (x - max_val).exp());
                let sum = lane.sum();
                lane.mapv_inplace(|x| x / sum);
            });
        }

        Tensor::new(new_data)
    }
}
