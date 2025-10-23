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
}
