"""
Inference module for fine-tuned language models.

Provides model loading, text generation with configurable sampling,
and inference speed benchmarking utilities.
"""

import logging
import time
from typing import Any, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation."""

    def __init__(self, stop_token_ids: list[int]):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class InferenceEngine:
    """
    Inference engine for loading and serving fine-tuned LLMs.

    Supports model loading from HuggingFace format, text generation
    with configurable decoding strategies, and throughput benchmarking.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_memory: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_memory = max_memory

        self.model = None
        self.tokenizer = None
        self.generation_config = None

    def load_model(self) -> None:
        """
        Load the fine-tuned model and tokenizer for inference.

        Configures device placement, precision, and memory mapping
        for optimal inference performance.
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        logger.info(f"Loading tokenizer from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model from: {self.model_path}")
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }

        if self.device == "auto":
            load_kwargs["device_map"] = "auto"
            if self.max_memory:
                load_kwargs["max_memory"] = self.max_memory
        else:
            load_kwargs["device_map"] = {"": self.device}

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        self.model.eval()

        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Model loaded: {param_count / 1e9:.2f}B parameters, "
            f"dtype={torch_dtype}"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_sequences: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Generate text from a prompt using the fine-tuned model.

        Returns the generated text along with metadata including
        token counts and generation timing.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens,
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        stopping_criteria = None
        if stop_sequences:
            stop_ids = []
            for seq in stop_sequences:
                token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
                if token_ids:
                    stop_ids.append(token_ids[-1])
            if stop_ids:
                stopping_criteria = StoppingCriteriaList(
                    [StopOnTokens(stop_ids)]
                )

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria,
            )

        elapsed = time.perf_counter() - start_time

        new_tokens = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        input_token_count = input_ids.shape[1]
        output_token_count = len(new_tokens)
        tokens_per_second = output_token_count / max(elapsed, 1e-6)

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "total_tokens": input_token_count + output_token_count,
            "generation_time_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_second, 1),
        }

    def benchmark(
        self,
        prompts: Optional[list[str]] = None,
        num_iterations: int = 10,
        max_new_tokens: int = 128,
        warmup_iterations: int = 2,
    ) -> dict[str, Any]:
        """
        Benchmark inference throughput and latency.

        Runs multiple generation passes and computes statistics
        on tokens/second, time-to-first-token, and latency distribution.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if prompts is None:
            prompts = [
                "Explain the concept of distributed computing in simple terms.",
                "What are the main differences between SQL and NoSQL databases?",
                "Describe the architecture of a modern web application.",
                "How does garbage collection work in Java?",
                "What is the CAP theorem and why does it matter?",
            ]

        logger.info(
            f"Starting benchmark: {num_iterations} iterations, "
            f"{max_new_tokens} max tokens, {warmup_iterations} warmup"
        )

        for i in range(warmup_iterations):
            prompt = prompts[i % len(prompts)]
            self.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        logger.info("Warmup complete")

        results = []
        for i in range(num_iterations):
            prompt = prompts[i % len(prompts)]
            result = self.generate(
                prompt, max_new_tokens=max_new_tokens, do_sample=False
            )
            results.append(result)

            logger.info(
                f"  Iteration {i + 1}/{num_iterations}: "
                f"{result['tokens_per_second']:.1f} tok/s, "
                f"{result['generation_time_seconds']:.3f}s"
            )

        times = [r["generation_time_seconds"] for r in results]
        throughputs = [r["tokens_per_second"] for r in results]
        output_tokens = [r["output_tokens"] for r in results]

        import statistics

        benchmark_results = {
            "num_iterations": num_iterations,
            "max_new_tokens": max_new_tokens,
            "latency": {
                "mean_seconds": round(statistics.mean(times), 4),
                "median_seconds": round(statistics.median(times), 4),
                "p95_seconds": round(sorted(times)[int(0.95 * len(times))], 4),
                "min_seconds": round(min(times), 4),
                "max_seconds": round(max(times), 4),
                "std_seconds": round(statistics.stdev(times) if len(times) > 1 else 0, 4),
            },
            "throughput": {
                "mean_tokens_per_sec": round(statistics.mean(throughputs), 1),
                "median_tokens_per_sec": round(statistics.median(throughputs), 1),
                "min_tokens_per_sec": round(min(throughputs), 1),
                "max_tokens_per_sec": round(max(throughputs), 1),
            },
            "tokens": {
                "mean_output_tokens": round(statistics.mean(output_tokens), 1),
                "total_output_tokens": sum(output_tokens),
            },
            "device": str(self.model.device),
            "dtype": self.torch_dtype,
            "model_path": self.model_path,
        }

        logger.info(
            f"Benchmark complete: "
            f"Avg {benchmark_results['throughput']['mean_tokens_per_sec']} tok/s, "
            f"Avg latency {benchmark_results['latency']['mean_seconds']}s"
        )

        return benchmark_results


def main():
    """Run inference with command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="LLM Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for generation")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device placement")
    args = parser.parse_args()

    engine = InferenceEngine(
        model_path=args.model_path,
        device=args.device,
    )
    engine.load_model()

    if args.benchmark:
        results = engine.benchmark()
        import json

        print(json.dumps(results, indent=2))
    elif args.prompt:
        result = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\nPrompt: {result['prompt']}")
        print(f"\nGenerated ({result['output_tokens']} tokens, {result['generation_time_seconds']}s):")
        print(result["generated_text"])
    else:
        print("Provide --prompt or --benchmark flag")


if __name__ == "__main__":
    main()
