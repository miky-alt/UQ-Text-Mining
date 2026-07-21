# --- Standard Library ---
import gc
import abc
from typing import Optional, Union, Any, Dict, List

# --- Third-Party Libraries ---
import pandas as pd
import numpy as np
import torch

# Assuming the Evaluator interface is located in the evaluation module
from .response_evaluator import BaseResponseEvaluator
from .uq_engine import evaluate_uncertainty

def _archive_metrics(
    registry: Dict[str, Dict[str, List[Any]]],
    tech: str,
    score: float,
    quality: float,
    prompt: str,
    response: str,
    question: str,
    ground_truth: str
) -> None:
    """Helper to archive extracted metrics into the provided registry."""
    registry[tech]["scores"].append(score)
    registry[tech]["qualities"].append(quality)
    registry[tech]["prompts"].append(prompt)
    registry[tech]["responses"].append(response)
    registry[tech]["questions"].append(question)
    registry[tech]["ground_truths"].append(ground_truth)

async def compute_dataset_uq_scores(
    dataset: pd.DataFrame,
    uq_techniques: List[str],
    uq_context: Any,
    evaluator: BaseResponseEvaluator,
    granularity: str = "sequence",
    tech_kwargs_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    **global_uq_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Executes sequential LLM inference and aggregates raw uncertainty metrics.
    Polymorphically handles both structured dictionary samples and raw text strings.
    """
    tech_kwargs_registry = tech_kwargs_registry or {}

    # Secure data ingestion from the data source
    if hasattr(dataset, "iterrows"):
        samples = [row.to_dict() for _, row in dataset.iterrows()]
    else:
        samples = [item for item in dataset]

    is_multimodal = hasattr(dataset, "image") or "image" in dataset.columns if hasattr(dataset, "columns") else False

    # Initialize the metrics storage registry
    raw_registry = {
        tech: {"scores": [], "qualities": [], "prompts": [], "responses": [], "questions": [], "ground_truths": []}
        for tech in uq_techniques
    }

    print(f"⏳ Extracting raw scores from {len(samples)} samples using {evaluator.__class__.__name__}...")
    
    for _, sample in enumerate(samples):
        # Defensive and polymorphic sample dispatching
        if isinstance(sample, dict):
            base_prompt = sample.get("question") or sample.get("prompt") or sample.get("text", "")
            true_answer = str(sample.get("answer", "N/A")).strip()
            image_payload = sample.get("image_base64")
        else:
            base_prompt = str(sample)
            true_answer = "N/A"
            image_payload = None

        prompt_input = f"<image>\n{base_prompt}" if (is_multimodal and image_payload) else base_prompt

        for tech in uq_techniques:
            # Enforce total memory isolation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            specific_tech_kwargs = tech_kwargs_registry.get(tech, {})
            merged_uq_kwargs = {**global_uq_kwargs, **specific_tech_kwargs}

            if image_payload:
                merged_uq_kwargs["image_base64"] = image_payload

            # Execute asynchronous inference
            result = await evaluate_uncertainty(
                prompt=prompt_input,
                technique_name=tech,
                granularity=granularity,
                uq_context=uq_context,
                **merged_uq_kwargs
            )

            raw_score = result.get("uncertainty_score", 0.0)
            generated_text = str(result.get("generated_text", "No response")).strip()

            # Compute quality using the injected evaluator
            is_correct = evaluator(generated_text=generated_text, ground_truth=true_answer)

            # Archive metadata and scores
            _archive_metrics(
                registry=raw_registry,
                tech=tech,
                score=raw_score,
                quality=is_correct,
                prompt=prompt_input,
                response=generated_text,
                question=base_prompt,
                ground_truth=true_answer
            )

    print("✅ Raw scores extraction completed successfully.")
    return raw_registry

async def compute_batch_uqlm_scores(
    prompts: List[str],
    reference_answers: List[str],
    uq_techniques: List[str],
    granularity: str,
    uq_context: Any,
    evaluator: BaseResponseEvaluator,
    model_alias: str = "default",
    batch_size: int = 16,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Standardized extraction utility specifically for UQLM frameworks.
    Strict Dependency Injection: expects a concrete BaseResponseEvaluator instance.
    """
    # 1. Initialize metrics storage registry
    raw_registry = {
        tech: {
            "scores": [],
            "qualities": [],
            "prompts": [],
            "responses": [],
            "questions": [],
            "ground_truths": []
        }
        for tech in uq_techniques
    }

    total_prompts = len(prompts)
    print(f"🚀 Starting extraction. Total: {total_prompts} samples | Batch: {batch_size} | Evaluator: {evaluator.__class__.__name__}")

    # 2. Process in batches
    for i in range(0, total_prompts, batch_size):
        end_idx = min(i + batch_size, total_prompts)
        chunk_prompts = prompts[i:end_idx]
        chunk_references = reference_answers[i:end_idx]

        for tech in uq_techniques:
            # PHASE 1: Raw Metrics Extraction
            batch_outputs = await evaluate_uncertainty(
                prompt=chunk_prompts,
                technique_name=tech,
                granularity=granularity,
                uq_context=uq_context,
                model_alias=model_alias,
                **kwargs
            )

            # PHASE 2: Processing and Grading
            for out, prompt, ref in zip(batch_outputs, chunk_prompts, chunk_references):
                raw_score = out.get("uncertainty_score", 0.0)
                generated_text = str(out.get("generated_text", "No response")).strip()

                # Use the injected evaluator instance
                is_correct = evaluator(generated_text=generated_text, ground_truth=ref)

                # PHASE 3: Archive to Registry
                _archive_metrics(
                    registry=raw_registry,
                    tech=tech,
                    score=raw_score,
                    quality=is_correct,
                    prompt=prompt,
                    response=generated_text,
                    question=prompt,
                    ground_truth=ref
                )

    print("✅ Raw scores extraction completed successfully.")
    return raw_registry