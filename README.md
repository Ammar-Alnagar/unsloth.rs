# Unsloth (Rust Version)

This is a Rust rewrite of the Unsloth framework, a tool for fine-tuning large language models.

## Building and Running

To build the project, run the following command:

```
cargo build --release
```

To run the command-line interface, use the following command:

```
cargo run --release -- --model <model_name>
```

## Features

* Data preparation pipeline
* Custom CUDA kernels for LoRA and QLoRA
* Llama model architecture
* Reinforcement learning with PPO
* Model saving and loading
* Training loop
* Command-line interface

## Future Work

* Implement the forward and backward passes for the custom kernels
* Implement the forward passes for the model architectures
* Implement the training logic for the PPO algorithm
* Implement the full model saving and loading functionality
* Add more command-line options and commands
* Write more comprehensive tests
