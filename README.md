# DeepSpeed LLM Fine-Tuning

Full-parameter fine-tuning of large language models using DeepSpeed ZeRO optimization. Supports multi-GPU distributed training with ZeRO Stage 2/3 for memory-efficient full fine-tuning of 7B+ parameter models.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                  Training Pipeline                     │
│                                                       │
│  ┌──────────┐    ┌──────────┐    ┌─────────────┐     │
│  │  Data     │───▶│ Training │───▶│ Checkpoint  │     │
│  │ Pipeline  │    │  Loop    │    │ Management  │     │
│  └──────────┘    └────┬─────┘    └──────┬──────┘     │
│                       │                  │            │
│                  ┌────▼─────┐    ┌──────▼──────┐     │
│                  │ DeepSpeed│    │  HuggingFace│     │
│                  │  Engine  │    │  Conversion │     │
│                  └────┬─────┘    └─────────────┘     │
│                       │                               │
│         ┌─────────────┼─────────────┐                │
│         │             │             │                │
│    ┌────▼───┐   ┌────▼───┐   ┌────▼───┐            │
│    │  GPU 0 │   │  GPU 1 │   │  GPU 2 │  ...       │
│    │ ZeRO   │   │ ZeRO   │   │ ZeRO   │            │
│    └────────┘   └────────┘   └────────┘            │
└───────────────────────────────────────────────────────┘
```

### ZeRO Optimization Stages

- **Stage 2**: Partitions optimizer states and gradients across GPUs. Each GPU holds full model parameters but shares optimizer memory. Best for 4-8 GPU setups.
- **Stage 3**: Partitions parameters, gradients, and optimizer states. Enables training models larger than single-GPU memory. Supports CPU offloading for further memory savings.

## Why DeepSpeed Over QLoRA

This project explores full-parameter fine-tuning as an alternative to parameter-efficient methods:

| Aspect | DeepSpeed (Full FT) | QLoRA (PEFT) |
|--------|---------------------|--------------|
| Parameters Trained | All (~7B) | ~0.1% (~10M) |
| GPU Requirements | 4-8x A100 80GB | 1x RTX 4090 24GB |
| Training Time | 8-24 hours | 2-6 hours |
| Model Quality | Highest possible | ~95-98% of full FT |
| Memory per GPU | ~20-40 GB (ZeRO-3) | ~12-18 GB |
| Infrastructure Cost | High | Low |
| Catastrophic Forgetting Risk | Higher | Lower |

**Decision**: QLoRA was chosen for the production deployment due to single-GPU accessibility, faster iteration cycles, and comparable performance on the target domain tasks (95%+ of full fine-tuning quality).

## Project Structure

```
deepspeed-llm-finetuning/
├── src/
│   ├── trainer.py           # DeepSpeed training loop
│   ├── data_pipeline.py     # Data loading and preprocessing
│   └── inference.py         # Model inference and benchmarking
├── configs/
│   ├── ds_config_zero2.json # DeepSpeed ZeRO Stage 2 config
│   ├── ds_config_zero3.json # DeepSpeed ZeRO Stage 3 config
│   └── training_config.yaml # Training hyperparameters
├── scripts/
│   └── train_distributed.sh # Distributed training launcher
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Single-Node Multi-GPU Training

```bash
# ZeRO Stage 2 (4 GPUs)
deepspeed --num_gpus=4 src/trainer.py \
    --config configs/training_config.yaml \
    --deepspeed configs/ds_config_zero2.json

# ZeRO Stage 3 with CPU offloading (4 GPUs)
deepspeed --num_gpus=4 src/trainer.py \
    --config configs/training_config.yaml \
    --deepspeed configs/ds_config_zero3.json
```

### Multi-Node Training

```bash
deepspeed --hostfile=hostfile.txt \
    --num_gpus=4 \
    --num_nodes=2 \
    src/trainer.py \
    --config configs/training_config.yaml \
    --deepspeed configs/ds_config_zero3.json
```

### Convert Checkpoint to HuggingFace Format

```python
from src.trainer import DeepSpeedFineTuner

trainer = DeepSpeedFineTuner(config_path="configs/training_config.yaml")
trainer.convert_to_hf(
    checkpoint_dir="checkpoints/epoch_3",
    output_dir="models/llama2-7b-finetuned"
)
```

## Configuration

### Training Config (`configs/training_config.yaml`)

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"

training:
  num_epochs: 3
  learning_rate: 2.0e-5
  warmup_ratio: 0.03
  max_seq_length: 2048
  gradient_checkpointing: true
```

### DeepSpeed Config

Choose ZeRO Stage 2 for faster training or Stage 3 for larger models:

- `ds_config_zero2.json`: Optimizer + gradient partitioning
- `ds_config_zero3.json`: Full parameter + optimizer + gradient partitioning with CPU offload

## Monitoring

Training metrics are logged to Weights & Biases:

```bash
export WANDB_PROJECT="deepspeed-finetuning"
export WANDB_API_KEY="your-key"
```

Tracked metrics: loss, learning rate, throughput (tokens/sec), GPU memory utilization, gradient norms.

## Tech Stack

- **Training**: PyTorch, DeepSpeed, Transformers
- **Data**: HuggingFace Datasets
- **Monitoring**: Weights & Biases
- **Evaluation**: ROUGE, Perplexity
- **Acceleration**: Mixed precision (BF16), gradient checkpointing
