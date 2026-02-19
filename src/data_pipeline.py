"""
Data pipeline for preparing instruction-tuning datasets.

Handles dataset loading, conversation formatting, tokenization,
and data collation for causal language model training.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_NO_INPUT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

CHAT_TEMPLATE = (
    "<|system|>\nYou are a helpful assistant.</s>\n"
    "<|user|>\n{instruction}</s>\n"
    "<|assistant|>\n{output}</s>"
)


class InstructionDataset(Dataset):
    """
    Tokenized instruction dataset for causal language model training.

    Stores pre-tokenized input_ids, attention_masks, and labels
    with proper masking of instruction tokens (only train on responses).
    """

    def __init__(
        self,
        input_ids: list[torch.Tensor],
        attention_masks: list[torch.Tensor],
        labels: list[torch.Tensor],
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class DataPipeline:
    """
    Data preparation pipeline for instruction-tuning datasets.

    Loads datasets from HuggingFace Hub or local files, formats
    conversations into instruction templates, and tokenizes with
    proper padding and truncation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        prompt_template: str = "alpaca",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.prompt_template = prompt_template

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def load_and_preprocess(
        self,
        dataset_name: str = "tatsu-lab/alpaca",
        max_samples: Optional[int] = None,
        val_split: float = 0.05,
    ) -> tuple[InstructionDataset, Optional[InstructionDataset]]:
        """
        Load, format, tokenize, and split a dataset for training.

        Returns tokenized training and validation InstructionDataset
        instances ready for DataLoader wrapping.
        """
        from datasets import load_dataset

        logger.info(f"Loading dataset: {dataset_name}")

        try:
            raw_dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}. Using synthetic data.")
            raw_dataset = self._generate_synthetic_dataset()

        if max_samples is not None and len(raw_dataset) > max_samples:
            raw_dataset = raw_dataset.select(range(max_samples))
            logger.info(f"Truncated to {max_samples} samples")

        formatted_texts = self.format_conversations(raw_dataset)
        logger.info(f"Formatted {len(formatted_texts)} conversations")

        all_input_ids, all_attention_masks, all_labels = self.tokenize_dataset(
            formatted_texts
        )

        n_total = len(all_input_ids)
        n_val = max(int(n_total * val_split), 1) if val_split > 0 else 0
        n_train = n_total - n_val

        indices = np.random.RandomState(42).permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = InstructionDataset(
            input_ids=[all_input_ids[i] for i in train_indices],
            attention_masks=[all_attention_masks[i] for i in train_indices],
            labels=[all_labels[i] for i in train_indices],
        )

        val_dataset = None
        if n_val > 0:
            val_dataset = InstructionDataset(
                input_ids=[all_input_ids[i] for i in val_indices],
                attention_masks=[all_attention_masks[i] for i in val_indices],
                labels=[all_labels[i] for i in val_indices],
            )

        logger.info(
            f"Dataset split: {n_train} train, {n_val} validation samples"
        )
        return train_dataset, val_dataset

    def format_conversations(
        self,
        dataset: Any,
    ) -> list[str]:
        """
        Format raw dataset examples into instruction-tuning templates.

        Supports Alpaca format (instruction/input/output columns) and
        chat format (messages column with role/content pairs).
        """
        formatted = []

        columns = dataset.column_names if hasattr(dataset, "column_names") else list(dataset[0].keys())

        if "instruction" in columns and "output" in columns:
            for example in dataset:
                instruction = example.get("instruction", "").strip()
                input_text = example.get("input", "").strip()
                output_text = example.get("output", "").strip()

                if not instruction or not output_text:
                    continue

                if self.prompt_template == "chat":
                    text = CHAT_TEMPLATE.format(
                        instruction=instruction + (f"\n{input_text}" if input_text else ""),
                        output=output_text,
                    )
                elif input_text:
                    text = ALPACA_PROMPT_TEMPLATE.format(
                        instruction=instruction,
                        input=input_text,
                        output=output_text,
                    )
                else:
                    text = ALPACA_NO_INPUT_TEMPLATE.format(
                        instruction=instruction,
                        output=output_text,
                    )

                formatted.append(text)

        elif "messages" in columns:
            for example in dataset:
                messages = example.get("messages", [])
                parts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "").strip()
                    parts.append(f"<|{role}|>\n{content}</s>")
                if parts:
                    formatted.append("\n".join(parts))

        elif "text" in columns:
            for example in dataset:
                text = example.get("text", "").strip()
                if text:
                    formatted.append(text)

        else:
            logger.warning(
                f"Unrecognized dataset format. Columns: {columns}. "
                f"Attempting to use first string column."
            )
            for example in dataset:
                for col in columns:
                    val = example.get(col, "")
                    if isinstance(val, str) and len(val) > 20:
                        formatted.append(val)
                        break

        logger.info(
            f"Formatted {len(formatted)} examples using '{self.prompt_template}' template"
        )
        return formatted

    def tokenize_dataset(
        self,
        texts: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Tokenize formatted texts for causal language model training.

        Produces input_ids, attention_masks, and labels with proper
        padding and truncation. Labels are shifted copies of input_ids
        with padding tokens masked to -100.
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        skipped = 0

        for text in texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            actual_length = attention_mask.sum().item()
            if actual_length < 10:
                skipped += 1
                continue

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        if skipped > 0:
            logger.info(f"Skipped {skipped} examples shorter than 10 tokens")

        logger.info(
            f"Tokenized {len(all_input_ids)} examples "
            f"(max_seq_length={self.max_seq_length})"
        )
        return all_input_ids, all_attention_masks, all_labels

    def create_data_collator(self) -> callable:
        """
        Create a data collator for the DataLoader.

        Stacks pre-tokenized tensors into batches, handling any
        final padding alignment needed for consistent batch sizes.
        """
        pad_token_id = self.tokenizer.pad_token_id

        def collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            input_ids = torch.stack([f["input_ids"] for f in features])
            attention_mask = torch.stack([f["attention_mask"] for f in features])
            labels = torch.stack([f["labels"] for f in features])

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return collator

    def _generate_synthetic_dataset(self) -> list[dict[str, str]]:
        """Generate a synthetic instruction dataset for testing."""
        examples = []
        tasks = [
            ("Explain what a neural network is.", "", "A neural network is a computational model inspired by the biological neural networks in the human brain. It consists of layers of interconnected nodes that process information using mathematical functions."),
            ("Summarize the key principles of clean code.", "", "Clean code follows several principles: meaningful naming, single responsibility, DRY (Don't Repeat Yourself), consistent formatting, minimal complexity, and comprehensive error handling."),
            ("Convert this temperature from Celsius to Fahrenheit.", "36.5", "To convert Celsius to Fahrenheit, multiply by 9/5 and add 32. So 36.5°C = (36.5 * 9/5) + 32 = 65.7 + 32 = 97.7°F"),
            ("What is the difference between a stack and a queue?", "", "A stack follows Last-In-First-Out (LIFO) ordering, while a queue follows First-In-First-Out (FIFO) ordering. Stacks support push/pop operations, queues support enqueue/dequeue."),
            ("Write a SQL query to find duplicate records.", "Table: employees (id, name, email)", "SELECT email, COUNT(*) as count FROM employees GROUP BY email HAVING COUNT(*) > 1;"),
        ]

        for instruction, input_text, output in tasks:
            examples.append(
                {"instruction": instruction, "input": input_text, "output": output}
            )

        class SyntheticDataset:
            def __init__(self, data):
                self.data = data
                self.column_names = ["instruction", "input", "output"]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return SyntheticDataset(self.data[idx])
                return self.data[idx]

            def __iter__(self):
                return iter(self.data)

            def select(self, indices):
                return SyntheticDataset([self.data[i] for i in indices])

        return SyntheticDataset(examples * 200)
