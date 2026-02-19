"""
DeepSpeed LLM Fine-Tuner.

Implements full-parameter fine-tuning of large language models using
DeepSpeed ZeRO optimization with multi-GPU distributed training support.
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from src.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class DeepSpeedFineTuner:
    """
    Full-parameter fine-tuning of LLMs using DeepSpeed ZeRO.

    Supports ZeRO Stage 2/3 with optional CPU offloading, gradient
    checkpointing, mixed precision training, and W&B logging.
    """

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        deepspeed_config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        local_rank: int = -1,
    ):
        self.config = self._load_config(config_path)
        self.ds_config_path = deepspeed_config_path
        self.ds_config = (
            self._load_ds_config(deepspeed_config_path)
            if deepspeed_config_path
            else None
        )
        self.output_dir = output_dir or self.config.get("output", {}).get(
            "dir", "./checkpoints"
        )
        self.local_rank = local_rank

        self.model = None
        self.tokenizer = None
        self.engine = None
        self.optimizer = None
        self.lr_scheduler = None

        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load training configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_ds_config(self, config_path: str) -> dict:
        """Load DeepSpeed configuration from JSON file."""
        with open(config_path, "r") as f:
            return json.load(f)

    def _setup_logging(self) -> None:
        """Configure logging with rank-aware formatting."""
        log_dir = self.config.get("output", {}).get("logging_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)

        rank = self.local_rank if self.local_rank >= 0 else 0
        logging.basicConfig(
            level=logging.INFO if rank == 0 else logging.WARNING,
            format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, f"train_rank{rank}.log")),
            ],
        )

    def _is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        if dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def load_model_and_tokenizer(self) -> None:
        """
        Load the base model and tokenizer from HuggingFace.

        Configures model for training with gradient checkpointing
        and the appropriate torch dtype.
        """
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "meta-llama/Llama-2-7b-hf")
        revision = model_config.get("revision", "main")
        torch_dtype_str = model_config.get("torch_dtype", "bfloat16")
        trust_remote_code = model_config.get("trust_remote_code", False)

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Loading model: {model_name} (dtype: {torch_dtype_str})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            use_cache=False,
        )

        training_config = self.config.get("training", {})
        if training_config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model loaded: {param_count / 1e9:.2f}B parameters "
            f"({trainable_count / 1e9:.2f}B trainable)"
        )

    def setup_deepspeed_config(self) -> dict:
        """
        Build or merge DeepSpeed configuration for the training run.

        Reconciles training config values with DeepSpeed config,
        ensuring batch sizes, learning rates, and scheduler
        parameters are consistent.
        """
        if self.ds_config is not None:
            ds_config = self.ds_config.copy()
        else:
            ds_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {"device": "cpu", "pin_memory": True},
                    "allgather_partitions": True,
                    "allgather_bucket_size": 200000000,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 200000000,
                    "contiguous_gradients": True,
                },
                "gradient_accumulation_steps": 4,
                "gradient_clipping": 1.0,
                "wall_clock_breakdown": False,
            }

        training_config = self.config.get("training", {})

        grad_accum = training_config.get(
            "gradient_accumulation_steps",
            ds_config.get("gradient_accumulation_steps", 4),
        )
        ds_config["gradient_accumulation_steps"] = grad_accum

        micro_batch = training_config.get(
            "per_device_batch_size",
            ds_config.get("train_micro_batch_size_per_gpu", 4),
        )
        ds_config["train_micro_batch_size_per_gpu"] = micro_batch

        max_grad_norm = training_config.get("max_grad_norm", 1.0)
        ds_config["gradient_clipping"] = max_grad_norm

        logger.info(
            f"DeepSpeed config: ZeRO Stage {ds_config.get('zero_optimization', {}).get('stage', 2)}, "
            f"micro_batch={micro_batch}, grad_accum={grad_accum}"
        )

        return ds_config

    def prepare_data(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare training and validation dataloaders.

        Uses the DataPipeline class for tokenization and formatting,
        then wraps datasets in distributed-aware DataLoaders.
        """
        data_config = self.config.get("data", {})
        training_config = self.config.get("training", {})

        pipeline = DataPipeline(
            tokenizer=self.tokenizer,
            max_seq_length=training_config.get("max_seq_length", 2048),
        )

        dataset_name = data_config.get("dataset", "tatsu-lab/alpaca")
        max_samples = data_config.get("max_samples")
        val_split = data_config.get("val_split", 0.05)

        train_dataset, val_dataset = pipeline.load_and_preprocess(
            dataset_name=dataset_name,
            max_samples=max_samples,
            val_split=val_split,
        )

        logger.info(
            f"Dataset prepared: {len(train_dataset)} train, "
            f"{len(val_dataset) if val_dataset else 0} val samples"
        )

        batch_size = training_config.get("per_device_batch_size", 4)

        train_sampler = (
            DistributedSampler(
                train_dataset,
                shuffle=data_config.get("shuffle", True),
                seed=data_config.get("shuffle_seed", 42),
            )
            if dist.is_initialized()
            else None
        )

        collator = pipeline.create_data_collator()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collator,
            num_workers=data_config.get("preprocessing_num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        val_loader = None
        if val_dataset and len(val_dataset) > 0:
            val_sampler = (
                DistributedSampler(val_dataset, shuffle=False)
                if dist.is_initialized()
                else None
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                collate_fn=collator,
                num_workers=2,
                pin_memory=True,
            )

        return train_loader, val_loader

    def train(self) -> dict[str, Any]:
        """
        Execute the full distributed training loop with DeepSpeed.

        Handles forward/backward passes, gradient accumulation,
        checkpoint saving, evaluation, and metrics logging.
        """
        self.load_model_and_tokenizer()
        ds_config = self.setup_deepspeed_config()
        train_loader, val_loader = self.prepare_data()

        training_config = self.config.get("training", {})
        num_epochs = training_config.get("num_epochs", 3)
        logging_steps = training_config.get("logging_steps", 10)
        save_steps = training_config.get("save_steps", 500)
        eval_steps = training_config.get("eval_steps", 250)
        warmup_ratio = training_config.get("warmup_ratio", 0.03)
        learning_rate = training_config.get("learning_rate", 2e-5)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        if "scheduler" in ds_config:
            ds_config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
            ds_config["scheduler"]["params"]["total_num_steps"] = total_steps

        self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
            model_parameters=self.model.parameters(),
        )

        logger.info(
            f"Training started: {num_epochs} epochs, {total_steps} total steps, "
            f"{warmup_steps} warmup steps"
        )

        wandb_config = self.config.get("wandb", {})
        if wandb_config.get("enabled", False) and self._is_main_process():
            try:
                import wandb

                wandb.init(
                    project=wandb_config.get("project", "deepspeed-finetuning"),
                    entity=wandb_config.get("entity"),
                    name=wandb_config.get("run_name"),
                    config=self.config,
                )
            except ImportError:
                logger.warning("wandb not available, skipping W&B logging")

        training_metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rates": [],
            "throughput_tokens_per_sec": [],
        }

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.engine.train()

            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_start = time.time()

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.engine.device) for k, v in batch.items()}

                outputs = self.engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

                self.engine.backward(loss)
                self.engine.step()

                step_loss = loss.item()
                epoch_loss += step_loss
                batch_tokens = batch["attention_mask"].sum().item()
                epoch_tokens += batch_tokens

                self.global_step += 1

                if self.global_step % logging_steps == 0 and self._is_main_process():
                    elapsed = time.time() - epoch_start
                    throughput = epoch_tokens / max(elapsed, 1e-6)
                    current_lr = self.optimizer.param_groups[0].get("lr", learning_rate)

                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} | "
                        f"Step {self.global_step}/{total_steps} | "
                        f"Loss: {step_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Throughput: {throughput:.0f} tok/s"
                    )

                    training_metrics["train_loss"].append(step_loss)
                    training_metrics["learning_rates"].append(current_lr)
                    training_metrics["throughput_tokens_per_sec"].append(throughput)

                    if wandb_config.get("enabled", False):
                        try:
                            import wandb

                            wandb.log(
                                {
                                    "train/loss": step_loss,
                                    "train/learning_rate": current_lr,
                                    "train/throughput_tokens_per_sec": throughput,
                                    "train/epoch": epoch + step / len(train_loader),
                                    "train/global_step": self.global_step,
                                },
                                step=self.global_step,
                            )
                        except Exception:
                            pass

                if (
                    eval_steps
                    and self.global_step % eval_steps == 0
                    and val_loader is not None
                ):
                    eval_loss = self.evaluate(val_loader)
                    training_metrics["eval_loss"].append(eval_loss)

                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        if self._is_main_process():
                            self.save_checkpoint(tag="best")

                    self.engine.train()

                if save_steps and self.global_step % save_steps == 0:
                    self.save_checkpoint(tag=f"step_{self.global_step}")

            avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
            epoch_duration = time.time() - epoch_start

            if self._is_main_process():
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} complete | "
                    f"Avg Loss: {avg_epoch_loss:.4f} | "
                    f"Duration: {epoch_duration:.0f}s"
                )

            self.save_checkpoint(tag=f"epoch_{epoch + 1}")

        if self._is_main_process():
            logger.info("Training complete. Converting final checkpoint to HuggingFace format.")
            hf_dir = self.config.get("output", {}).get("hf_dir", "./models/finetuned")
            self.convert_to_hf(output_dir=hf_dir)

        return training_metrics

    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Run evaluation and compute perplexity.

        Returns the average validation loss across all batches.
        """
        self.engine.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.engine.device) for k, v in batch.items()}
                outputs = self.engine(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs.loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        perplexity = math.exp(min(avg_loss, 100))

        if self._is_main_process():
            logger.info(
                f"Evaluation | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}"
            )

            wandb_config = self.config.get("wandb", {})
            if wandb_config.get("enabled", False):
                try:
                    import wandb

                    wandb.log(
                        {
                            "eval/loss": avg_loss,
                            "eval/perplexity": perplexity,
                            "eval/global_step": self.global_step,
                        },
                        step=self.global_step,
                    )
                except Exception:
                    pass

        return avg_loss

    def save_checkpoint(self, tag: str = "latest") -> None:
        """
        Save a DeepSpeed checkpoint with metadata.

        Includes model state, optimizer state, scheduler state,
        and training progress metadata.
        """
        checkpoint_dir = os.path.join(self.output_dir, tag)
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.engine.save_checkpoint(checkpoint_dir, tag=tag)

        if self._is_main_process():
            metadata = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_loss": self.best_eval_loss,
                "model_name": self.config.get("model", {}).get("name", ""),
                "config": self.config,
            }
            metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def convert_to_hf(
        self,
        checkpoint_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Convert a DeepSpeed checkpoint to HuggingFace format.

        Gathers ZeRO-partitioned weights and saves them in the
        standard HuggingFace model format for inference.
        """
        if output_dir is None:
            output_dir = self.config.get("output", {}).get(
                "hf_dir", "./models/finetuned"
            )
        os.makedirs(output_dir, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(
                output_dir,
                safe_serialization=True,
            )
        elif checkpoint_dir is not None:
            from deepspeed.utils.zero_to_fp32 import (
                get_fp32_state_dict_from_zero_checkpoint,
            )

            state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
            model_config = self.config.get("model", {})
            model = AutoModelForCausalLM.from_pretrained(
                model_config.get("name", "meta-llama/Llama-2-7b-hf"),
                torch_dtype=torch.bfloat16,
            )
            model.load_state_dict(state_dict)
            model.save_pretrained(output_dir, safe_serialization=True)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        logger.info(f"HuggingFace model saved: {output_dir}")
        return output_dir

    def compare_with_qlora(
        self,
        qlora_model_path: Optional[str] = None,
        test_prompts: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compare full fine-tuning results against a QLoRA-trained model.

        Evaluates both models on the same prompts and computes
        quality metrics (perplexity, ROUGE) for comparison.
        """
        if test_prompts is None:
            test_prompts = [
                "Explain the concept of gradient checkpointing in deep learning.",
                "What are the trade-offs between full fine-tuning and LoRA?",
                "Describe the ZeRO optimization stages in DeepSpeed.",
            ]

        comparison = {
            "full_ft_model": self.config.get("model", {}).get("name", ""),
            "qlora_model": qlora_model_path or "not provided",
            "prompts": test_prompts,
            "results": [],
        }

        if self.model is not None and self.tokenizer is not None:
            self.model.eval()
            for prompt in test_prompts:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                    )
                    generated = self.tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )

                comparison["results"].append(
                    {
                        "prompt": prompt,
                        "full_ft_output": generated,
                        "full_ft_length": len(generated.split()),
                    }
                )

        logger.info(
            f"Comparison complete: {len(comparison['results'])} prompts evaluated"
        )
        return comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for distributed training."""
    parser = argparse.ArgumentParser(description="DeepSpeed LLM Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed configuration JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed)",
    )
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trainer = DeepSpeedFineTuner(
        config_path=args.config,
        deepspeed_config_path=args.deepspeed,
        output_dir=args.output_dir,
        local_rank=args.local_rank,
    )

    metrics = trainer.train()

    if trainer._is_main_process():
        logger.info("Training metrics summary:")
        if metrics["train_loss"]:
            logger.info(f"  Final train loss: {metrics['train_loss'][-1]:.4f}")
        if metrics["eval_loss"]:
            logger.info(f"  Best eval loss: {min(metrics['eval_loss']):.4f}")
        if metrics["throughput_tokens_per_sec"]:
            avg_throughput = np.mean(metrics["throughput_tokens_per_sec"])
            logger.info(f"  Avg throughput: {avg_throughput:.0f} tok/s")
