use unsloth_rs::dataprep::synthetic::SyntheticDataKit;

#[test]
fn test_create_synthetic_data_kit() {
    let synthetic_data_kit = SyntheticDataKit::new();
    synthetic_data_kit.prepare_qa_generation();
}
