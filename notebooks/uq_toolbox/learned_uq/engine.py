from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .manager import SupervisedUQManager


def _render_prompt(
    prompt: str,
    manager: SupervisedUQManager,
    system_prompt: Optional[str] = None,
) -> str:
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    if getattr(manager.tokenizer, "chat_template", None):
        return manager.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if system_prompt:
        return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

    return prompt


def _extract_token_uncertainty(output: Dict[str, Any]) -> np.ndarray:
    if "uncertainty_score" not in output:
        raise KeyError(
            "Output does not contain 'uncertainty_score'. "
            f"Available keys: {list(output.keys())}"
        )

    uncertainty = output["uncertainty_score"]

    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.detach().float().cpu().numpy()

    uncertainty = np.asarray(uncertainty, dtype=float)

    while uncertainty.ndim > 1:
        uncertainty = uncertainty[0]

    uncertainty = uncertainty.reshape(-1)
    uncertainty = uncertainty[np.isfinite(uncertainty)]

    if uncertainty.size == 0:
        raise ValueError("The uncertainty head returned no finite values.")

    return uncertainty


def evaluate_supervised_uncertainty(
    prompt: str,
    manager: SupervisedUQManager,
    *,
    label: Optional[str] = None,
    system_prompt: Optional[str] = None,
    aggregation: str = "mean",
    max_input_length: Optional[int] = None,
) -> Dict[str, Any]:
    if not manager.is_loaded or manager.adapter is None:
        raise RuntimeError(
            "The supervised manager is not loaded. Call manager.load() first."
        )

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    chat_text = _render_prompt(
        prompt=prompt,
        manager=manager,
        system_prompt=system_prompt,
    )

    inputs = manager.tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length or manager.max_input_length,
        add_special_tokens=False,
    )

    input_ids = inputs["input_ids"].to(manager.input_device)

    with torch.inference_mode():
        output = manager.adapter.generate(input_ids)

    if "sequences" not in output:
        raise KeyError(
            "The supervised backend did not return generated sequences."
        )

    sequences = output["sequences"]
    generated_ids = sequences[:, input_ids.shape[1]:]

    response = manager.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    token_uncertainty = _extract_token_uncertainty(output)

    aggregators = {
        "mean": np.mean,
        "max": np.max,
        "median": np.median,
        "sum": np.sum,
    }

    if aggregation not in aggregators:
        raise ValueError(
            "aggregation must be one of: 'mean', 'max', 'median', or 'sum'."
        )

    sequence_uncertainty = float(
        aggregators[aggregation](token_uncertainty)
    )

    token_ids = generated_ids[0].detach().cpu().tolist()
    generated_tokens = manager.tokenizer.convert_ids_to_tokens(token_ids)

    aligned_length = min(
        len(generated_tokens),
        len(token_uncertainty),
    )

    return {
        "prompt": label or prompt,
        "input_prompt": prompt,
        "response": response,
        "generated_text": response,
        "supervised_uncertainty": sequence_uncertainty,
        "uncertainty_score": sequence_uncertainty,
        "token_uncertainties": token_uncertainty[:aligned_length].tolist(),
        "generated_tokens": generated_tokens[:aligned_length],
        "aggregation": aggregation,
        "library": "supervised_uq",
        "estimator_name": "learned_uncertainty_head",
        "granularity": "sequence",
        "model_name": manager.model_name,
        "uncertainty_head_name": manager.uncertainty_head_name,
    }


def evaluate_supervised_batch(
    prompts: Sequence[str],
    manager: SupervisedUQManager,
    *,
    labels: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
    aggregation: str = "mean",
    max_input_length: Optional[int] = None,
    continue_on_error: bool = False,
) -> List[Dict[str, Any]]:
    if labels is not None and len(labels) != len(prompts):
        raise ValueError("labels and prompts must have equal lengths.")

    results = []

    for index, prompt in enumerate(prompts):
        label = labels[index] if labels is not None else None

        try:
            result = evaluate_supervised_uncertainty(
                prompt=prompt,
                label=label,
                manager=manager,
                system_prompt=system_prompt,
                aggregation=aggregation,
                max_input_length=max_input_length,
            )
            result["sample_index"] = index
            results.append(result)

        except Exception as error:
            if not continue_on_error:
                raise

            results.append(
                {
                    "sample_index": index,
                    "prompt": label or prompt,
                    "input_prompt": prompt,
                    "response": "",
                    "generated_text": "",
                    "supervised_uncertainty": None,
                    "uncertainty_score": None,
                    "token_uncertainties": [],
                    "generated_tokens": [],
                    "error": str(error),
                    "library": "supervised_uq",
                    "estimator_name": "learned_uncertainty_head",
                    "granularity": "sequence",
                }
            )

    return results
