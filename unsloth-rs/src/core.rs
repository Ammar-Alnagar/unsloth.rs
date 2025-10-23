#[derive(Debug)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }

    pub fn get_data(&self) -> &Vec<f32> {
        &self.data
    }
}
