use ndarray::array;
use unsloth_rs::core::Tensor;

#[test]
fn test_rmsnorm() {
    let input_data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    let weight_data = array![0.1, 0.2, 0.3].into_dyn();
    let input = Tensor::new(input_data);
    let weight = Tensor::new(weight_data);
    let epsilon = 1e-5;

    let result = input.rmsnorm(&weight, epsilon);

    let expected_data = array![
        [0.046291, 0.18516402, 0.41661902],
        [0.07895449, 0.19738622, 0.3552952]
    ]
    .into_dyn();
    let expected = Tensor::new(expected_data);

    let diff = &result.data - &expected.data;
    let max_abs_diff = diff.mapv(f32::abs).iter().fold(0.0, |max, &val| val.max(max));
    assert!(max_abs_diff < 1e-5, "RMSNorm test failed. Max diff: {}", max_abs_diff);
}

#[test]
fn test_rope() {
    let input_data = array![
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ].into_dyn();
    let input = Tensor::new(input_data);

    let result = input.rope(1, 4, 10, 10000.0);

    let expected_data = array![
        [-1.1426396, 1.9220755, 2.9598508, 4.0297995],
        [-2.3473144, 7.449168, 6.9196515, 8.069599],
    ].into_dyn();
    let expected = Tensor::new(expected_data);

    let diff = &result.data - &expected.data;
    let max_abs_diff = diff.mapv(f32::abs).iter().fold(0.0, |max, &val| val.max(max));
    assert!(max_abs_diff < 1e-5, "RoPE test failed");
}

#[test]
fn test_softmax() {
    let input_data = array![1.0, 2.0, 3.0].into_dyn();
    let input = Tensor::new(input_data);

    let result = input.softmax(0);

    let expected_data = array![0.09003057, 0.24472847, 0.66524096].into_dyn();
    let expected = Tensor::new(expected_data);

    let diff = &result.data - &expected.data;
    let max_abs_diff = diff.mapv(f32::abs).iter().fold(0.0, |max, &val| val.max(max));
    assert!(max_abs_diff < 1e-6, "Softmax test failed");
}
