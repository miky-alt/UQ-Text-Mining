# --- Standard Library ---
import gc
import abc
from typing import Optional, Union, Any, Dict, List
import copy
# --- Third-Party Libraries ---
import pandas as pd
import numpy as np
import torch
# --- Local Modules ---
from .response_evaluator import BaseResponseEvaluator
from .uq_engine import evaluate_uncertainty
from uqlm.utils.results import UQResult



class UQResultContainer(dict):
    """
    Container ibrido avanzato che intercetta l'assegnazione dei dati,
    converte rigorosamente gli score in float e preserva integralmente
    sia le metriche di UQ che i metadati (responses, qualities, ecc.) 
    richiesti dai moduli di valutazione nativi di uqlm.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sanitize_scores()

    def __setitem__(self, key, value):
        if isinstance(value, dict) and "scores" in value:
            value["scores"] = [float(s) for s in value["scores"]]
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._sanitize_scores()

    def _sanitize_scores(self):
        for tech, metrics in self.items():
            if isinstance(metrics, dict) and "scores" in metrics:
                metrics["scores"] = [float(s) for s in metrics["scores"]]

    @property
    def data(self) -> Dict[str, Any]:
        self._sanitize_scores()
        result_data = {}
        for key, value in self.items():
            if isinstance(value, dict) and "scores" in value:
                result_data[key] = value["scores"]
            else:
                result_data[key] = value
        return result_data

    def to_df(self) -> pd.DataFrame:
        self._sanitize_scores()
        df_data = {}
        
        for tech, metrics in self.items():
            if isinstance(metrics, dict):
                if "scores" in metrics:
                    df_data[tech] = metrics["scores"]
                if "qualities" in metrics and "response_correct" not in df_data:
                    df_data["response_correct"] = metrics["qualities"]
                if "responses" in metrics and "response" not in df_data:
                    df_data["response"] = metrics["responses"]
                if "ground_truths" in metrics and "ground_truth" not in df_data:
                    df_data["ground_truth"] = metrics["ground_truths"]
            elif tech in ["responses", "qualities", "ground_truths", "response_correct"]:
                if tech == "responses":
                    df_data["response"] = metrics
                elif tech in ["qualities", "response_correct"]:
                    df_data["response_correct"] = metrics
                elif tech == "ground_truths":
                    df_data["ground_truth"] = metrics
                    
        return pd.DataFrame(df_data)

def _archive_metrics(
    registry: Dict[str, Dict[str, List[Any]]],
    tech: str,
    score: float,
    quality: float,
    prompt: str,
    response: str,
    question: str,
    ground_truth: str,
    uqlm_result: Optional[Any] = None  #
) -> None:
    """Helper to archive extracted metrics into the provided registry."""
    registry[tech]["scores"].append(score)
    registry[tech]["qualities"].append(quality)
    registry[tech]["prompts"].append(prompt)
    registry[tech]["responses"].append(response)
    registry[tech]["questions"].append(question)
    registry[tech]["ground_truths"].append(ground_truth)
    
    if uqlm_result is not None and "uqlm_result" in registry[tech]:
        if isinstance(registry[tech]["uqlm_result"], list):
            registry[tech]["uqlm_result"].append(uqlm_result)
        else:
            registry[tech]["uqlm_result"] = uqlm_result




async def compute_dataset_uq_scores(
    dataset: pd.DataFrame,
    uq_techniques: List[str],
    uq_models: Any,
    evaluator: BaseResponseEvaluator,
    granularity: str = "sequence",
    tech_kwargs_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    **global_uq_kwargs
) -> UQResultContainer:
    """
    Executes sequential LLM inference and aggregates raw uncertainty metrics.
    Returns a UQResultContainer for seamless integration with ScoreCalibrator.
    """
    tech_kwargs_registry = tech_kwargs_registry or {}

    # Secure data ingestion from the data source
    if hasattr(dataset, "iterrows"):
        samples = [row.to_dict() for _, row in dataset.iterrows()]
    else:
        samples = [item for item in dataset]

    is_multimodal = hasattr(dataset, "image") or ("image" in dataset.columns if hasattr(dataset, "columns") else False)

    # Initialize the metrics storage registry using the hybrid container
    raw_registry = UQResultContainer({
        tech: {"scores": [], "qualities": [], "prompts": [], "responses": [], "questions": [], "ground_truths": []}
        for tech in uq_techniques
    })

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
                uq_models=uq_models,
                **merged_uq_kwargs
            )

            raw_score = result.get("uncertainty_score", 0.0)
            generated_text = str(result.get("generated_text", "No response")).strip()

            # Compute quality using the injected evaluator polymorphically
            is_correct = await evaluator(
                prompt=prompt_input,
                generated_text=generated_text,
                ground_truth=true_answer
            )

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
    dataset: pd.DataFrame,
    uq_techniques: List[str],
    uq_models: Any,
    evaluator: BaseResponseEvaluator,
    granularity: str = "sequence",
    batch_size: int = 16,
    tech_kwargs_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    **global_uq_kwargs
) -> UQResultContainer:
    """
    Standardized high-throughput extraction utility returning a single 
    globally merged UQResultContainer complete with a unified uqlm_result UQResult object.
    """
    tech_kwargs_registry = tech_kwargs_registry or {}

    # Data Ingestion
    samples = [row.to_dict() for _, row in dataset.iterrows()] if hasattr(dataset, "iterrows") else [item for item in dataset]
    is_multimodal = any(c in dataset.columns for c in ["image", "image_base64"]) if hasattr(dataset, "columns") else False

    total_samples = len(samples)
    print(f"⏳ Extracting scores from {total_samples} samples across batches (Batch size: {batch_size}) using {evaluator.__class__.__name__}...")

    # Inizializziamo il container globale unificato
    global_registry = UQResultContainer({
        tech: {
            "scores": [],
            "qualities": [],
            "prompts": [],
            "responses": [],
            "questions": [],
            "ground_truths": [],
            "uqlm_result": None  # Verrà popolato dall'oggetto UQResult unito finale
        }
        for tech in uq_techniques
    })

    # Dizionario temporaneo per raccogliere i singoli uqlm_result batch per batch
    batch_uqlm_results_collection = {tech: [] for tech in uq_techniques}

    # Iterazione per batch
    for start_idx in range(0, total_samples, batch_size):
        batch_samples = samples[start_idx : start_idx + batch_size]
        
        batch_prompts = []
        batch_metadata = []
        
        for sample in batch_samples:
            base_prompt = sample.get("question") or sample.get("prompt") or sample.get("text", "")
            true_answer = str(sample.get("answer", "N/A")).strip()
            image_payload = sample.get("image_base64") or sample.get("image")
            
            prompt_input = f"<image>\n{base_prompt}" if (is_multimodal and image_payload) else base_prompt
            
            batch_prompts.append(prompt_input)
            batch_metadata.append({
                "base_prompt": base_prompt,
                "true_answer": true_answer,
                "image_payload": image_payload
            })

        # Iteriamo sulle tecniche per il batch corrente
        for tech in uq_techniques:
            gc.collect()
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()

            merged_uq_kwargs = {**global_uq_kwargs, **tech_kwargs_registry.get(tech, {})}
            if any(m["image_payload"] for m in batch_metadata):
                merged_uq_kwargs["image_base64"] = batch_metadata[0]["image_payload"]

            # Chiamata Batch nativa via lista di prompt
            results = await evaluate_uncertainty(
                prompt=batch_prompts, 
                technique_name=tech,
                granularity=granularity,
                uq_models=uq_models,
                **merged_uq_kwargs
            )

            if not isinstance(results, list): 
                results = [results]

            # Catturiamo il raw_uq_result del batch e lo accumuliamo nella collection separata
            if results:
                batch_uqlm_res = results[0].get("raw_uq_result")
                if batch_uqlm_res is not None:
                    batch_uqlm_results_collection[tech].append(batch_uqlm_res)

            # Popolamento dei dati del batch e archiviazione nel container globale
            for i, result in enumerate(results):
                raw_score = result.get("uncertainty_score", 0.0)
                generated_text = str(result.get("generated_text", "No response")).strip()
                meta = batch_metadata[i]

                # Valutazione della correttezza
                is_correct = await evaluator(
                    prompt=batch_prompts[i],
                    generated_text=generated_text,
                    ground_truth=meta["true_answer"]
                )

                # Archiviazione nel global_registry
                _archive_metrics(
                    registry=global_registry,
                    tech=tech,
                    score=raw_score,
                    quality=is_correct,
                    prompt=batch_prompts[i],
                    response=generated_text,
                    question=meta["base_prompt"],
                    ground_truth=meta["true_answer"],
                    uqlm_result=None
                )

    # =========================================================================
    # ACCORPAMENTO FINALE DEGLI OGGETTI UQRESULT DEI VARI BATCH (Via API UQResult)
    # =========================================================================
    for tech in uq_techniques:
        batch_results_list = batch_uqlm_results_collection[tech]
        if batch_results_list:
            batch_dicts = [
                res.to_dict() if hasattr(res, "to_dict") else dict(res) 
                for res in batch_results_list
            ]
            
            if batch_dicts:
                merged_dict = copy.deepcopy(batch_dicts[0])
                
                if "data" in merged_dict and isinstance(merged_dict["data"], dict):
                    for key in merged_dict["data"].keys():
                        for b_dict in batch_dicts[1:]:
                            if "data" in b_dict and key in b_dict["data"]:
                                merged_dict["data"][key].extend(b_dict["data"][key])
                
                # Ricreiamo un unico UQResult globale pulito
                global_registry[tech]["uqlm_result"] = UQResult(merged_dict)

    print("✅ Batch-wise raw scores extraction and UQResult global merging completed successfully.")
    return global_registry
