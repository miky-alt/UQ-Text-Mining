import getpass
import os
import base64
import torch
import numpy as np
from typing import Optional, Union, Any, List
# --- lm_polygraph Framework Imports ---
from lm_polygraph import estimate_uncertainty
from lm_polygraph.model_adapters import VisualWhiteboxModel
from lm_polygraph.stat_calculators import ClaimsExtractor, GreedyProbsCalculator
from lm_polygraph.stat_calculators.greedy_alternatives_nli import (
    GreedyAlternativesNLICalculator,
)
from lm_polygraph.stat_calculators.greedy_visual_probs import (
    GreedyProbsVisualCalculator,
)
from lm_polygraph.stat_calculators.prompt_visual import (
    ClaimPromptVisualCalculator,
)
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.utils.model import BlackboxModel, WhiteboxModel
from lm_polygraph.utils.openai_chat import OpenAIChat

# --- LangChain / UQLM Driver Imports ---
from langchain_core.messages import HumanMessage
from uqlm import (
    BlackBoxUQ,
    UQEnsemble,
    WhiteBoxUQ,
)

from ..registry import UQ_REGISTRY

from ..registry import UQLibrary



# =====================================================================
# 1. LM_POLYGRAPH HELPER FUNCTIONS (Sub-Engine Logic)
# =====================================================================

def _evaluate_claim_level_polygraph(
    prompt: str,
    estimator_class: Any,
    model: Any,
    image_path: Optional[str] = None
) -> dict:
    """
    Handles the multi-step pipeline for Claim-level Uncertainty Quantification.
    Executes dynamic routing to text or visual calculators based on context metadata.
    """
    mode_str = "MULTIMODAL" if image_path else "TEXT"
    print(f"   ↳ Initializing Claim-Level Pipeline ({mode_str} Mode)...")

    # Verify OpenAI API key validity for GPT-driven atomic claim extraction
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n   ⚠️ Warning: Missing OpenAI API Key for ClaimsExtractor.")
        api_key = getpass.getpass("    Enter your OpenAI API Key (sk-...): ")
        os.environ["OPENAI_API_KEY"] = api_key.strip()
        print("   OpenAI API key configured successfully.\n")

    deps = {}
    batch_texts = [prompt]

    if image_path:
        deps["images"] = [image_path]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Phase 1: Text generation and raw logit probability extraction
    print("   ↳ Step 1: Generating text and computing logit probabilities...")
    greedy_calc = GreedyProbsVisualCalculator() if image_path else GreedyProbsCalculator()
    deps.update(greedy_calc(deps, texts=batch_texts, model=model))

    # Phase 2: Atomic sub-claims breakdown via LLM parsing
    print("   ↳ Step 2: Extracting atomic claims (ClaimsExtractor via GPT-4o)...")
    extractor = ClaimsExtractor(OpenAIChat("gpt-4o"))
    deps.update(extractor(deps, texts=batch_texts, model=model))

    # Phase 3: Factual alignment evaluation (DeBERTa NLI vs Visual Prompt Grounding)
    print("   ↳ Step 3: Evaluating claim consistency and truthfulness...")
    judge_calc = (
        ClaimPromptVisualCalculator()
        if image_path
        else GreedyAlternativesNLICalculator(Deberta(device=device))
    )
    deps.update(judge_calc(deps, texts=batch_texts, model=model))

    # Phase 4: Compute targeted uncertainty score mapping
    print(f"   ↳ Step 4: Computing uncertainty metrics via {estimator_class.__class__.__name__}...")
    deps["model"] = model
    claim_scores = estimator_class(deps)

    # Phase 5: Build unified output metadata schema
    print("   ↳ Step 5: Formatting final output payload...")
    claims_list = deps["claims"][0]
    scores_list = claim_scores[0]

    claim_details = []
    for claim_obj, score in zip(claims_list, scores_list):
        claim_details.append({
            "claim_text": claim_obj.claim_text,
            "score": float(score)
        })

    return {
        "input_prompt": prompt,
        "image_used": image_path if image_path else "None",
        "generated_text": deps["greedy_texts"][0],
        "uncertainty_score": claim_details,
    }


def _handle_polygraph_execution(
    prompt: str,
    tech_info: dict,
    granularity: str,
    polygraph_model: Any,
    image_path: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Orchestrates validation execution and parsing routines for the
    lm_polygraph backend layer.
    """
    estimator_type = tech_info["estimator_class"]
    estimator = (
        estimator_type(**kwargs)
        if isinstance(estimator_type, type)
        else estimator_type
    )
    estimator_name = getattr(
        estimator_type, "__name__", estimator.__class__.__name__
    )

    # ⚠️ FIX: Detect prompt modality status (Multimodal vs. Base Text)
    is_multimodal = isinstance(prompt, str) and "<image>" in prompt
    clean_prompt = (
        prompt.replace("<image>", "").strip()
        if isinstance(prompt, str)
        else prompt
    )

    execution_args = {"input_text": clean_prompt}

    # The image asset is passed to Polygraph ONLY if the current prompt explicitly requires it
    if image_path and is_multimodal:
        execution_args["input_image"] = image_path

    # --- Granularity Level: SEQUENCE ---
    if granularity == "sequence":
        output = estimate_uncertainty(polygraph_model, estimator, **execution_args)
        raw_score = output.uncertainty
        final_score = (
            float(raw_score[0])
            if isinstance(raw_score, (np.ndarray, list))
            else float(raw_score)
        )

        return {
            "library": "lm_polygraph",
            "estimator_name": output.estimator,
            "granularity": granularity,
            "input_prompt": prompt,
            "generated_text": output.generation_text,
            "uncertainty_score": final_score,
        }

    # --- Granularity Level: CLAIM ---
    elif granularity == "claim":
        # Forward the image to the sub-engine only if validated by the <image> token
        result_payload = _evaluate_claim_level_polygraph(
            clean_prompt,
            estimator,
            polygraph_model,
            image_path=image_path if is_multimodal else None,
        )
        result_payload.update({
            "library": "lm_polygraph",
            "estimator_name": estimator_name,
            "granularity": granularity,
            "input_prompt": prompt,
        })
        return result_payload

    # --- Granularity Level: TOKEN ---
    elif granularity == "token":
        output = estimate_uncertainty(polygraph_model, estimator, **execution_args)
        raw_score = output.uncertainty
        if not isinstance(raw_score, (list, np.ndarray)):
            raise ValueError(
                f"The metric '{estimator_name}' returns an aggregate sequence score and cannot be mapped word-by-word."
            )

        clean_score_list = (
            raw_score.tolist() if isinstance(raw_score, np.ndarray) else list(raw_score)
        )
        raw_tokens = (
            output.generation_tokens[0]
            if isinstance(output.generation_tokens[0], list)
            else output.generation_tokens
        )

        if raw_tokens and isinstance(raw_tokens[0], int):
            token_strings = polygraph_model.tokenizer.convert_ids_to_tokens(raw_tokens)
        else:
            token_strings = raw_tokens

        token_details = [
            {
                "token": str(t).replace("▁", " ").replace("Ġ", " ").replace(" ", " ").strip(),
                "score": float(s),
            }
            for t, s in zip(token_strings, clean_score_list)
        ]

        return {
            "library": "lm_polygraph",
            "estimator_name": output.estimator,
            "granularity": granularity,
            "input_prompt": prompt,
            "generated_text": output.generation_text,
            "uncertainty_score": token_details,
        }

    raise ValueError(
        f"Granularity level '{granularity}' is not supported by lm_polygraph."
    )


# =====================================================================
# 2. MULTIMODAL PROMPT MAPPING LAYER (UQLM Helpers)
# =====================================================================

def _prepare_uqlm_execution_prompts(
    prompts_list: List[Union[str, Any]],
    image_base64: Optional[str]
) -> List[Any]:
    """
    Maps raw prompt strings and optional base64 image data into unified execution formats.
    Wraps inputs in LangChain HumanMessage schemas only if the prompt explicitly contains
    the '<image>' target token and valid visual bytes are provided.
    """
    execution_prompts = []

    for p in prompts_list:
        if isinstance(p, str):
            clean_text = p.replace("<image>", "").strip()

            # Strict Condition: Attach OpenAI Vision dictionary structure only for multimodal prompts
            if image_base64 and "<image>" in p:
                execution_prompts.append([
                    HumanMessage(
                        content=[
                            {"type": "text", "text": clean_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"},},])])
            else:
                # Base Prompt or Text-Only: Linear fallback to raw string input
                execution_prompts.append(clean_text)
        else:
            # Safeguard for message objects already pre-compiled upstream
            execution_prompts.append(p)

    return execution_prompts


# =====================================================================
# 3. CENTRAL UQLM ROUTING EXECUTOR
# =====================================================================

async def _handle_uqlm_execution(
    prompt: Union[str, List[str]],
    technique_name: str,
    tech_info: dict,
    granularity: str,
    uq_engine: Any,
    model_alias: str = "default",
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    **kwargs,
) -> Union[dict, List[dict]]:
    """
    Handles execution pipeline constraints, White/Black box environment compliance audits,
    and structured matrix harvesting for single scorers and multi-component UQLM ensembles.
    """
    if model_alias not in uq_engine.langchain_llms:
        raise KeyError(
            f"❌ Execution Error: No UQLM model registered under alias '{model_alias}'."
        )

    langchain_llm = uq_engine.langchain_llms[model_alias]
    uqlm_class = (
        tech_info["wrapper_class"][granularity]
        if isinstance(tech_info["wrapper_class"], dict)
        else tech_info["wrapper_class"]
    )

    # Dynamically initialize the UQ Wrapper (Ensemble vs. Single Scorer)
    if issubclass(uqlm_class, UQEnsemble):
        config_path = kwargs.pop("ensemble_config_path", None)
        if config_path:
            uqlm_wrapper = uqlm_class.load_config(config_path, llm=langchain_llm)
        else:
            scorers_list = kwargs.pop("ensemble_scorers", None)
            uqlm_wrapper = uqlm_class(
                llm=langchain_llm,
                scorers=scorers_list,
                max_calls_per_min=1000,
                device="cpu",
                nli_model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                **kwargs,
            )
        result_key = "ensemble_scores"
    else:
        real_scorer_name = tech_info.get(
            "scorer_id", technique_name.replace("uqlm_", "")
        )
        uqlm_wrapper = uqlm_class(
            llm=langchain_llm, scorers=[real_scorer_name], **kwargs
        )
        result_key = real_scorer_name

    # Normalize input into an iterable list format
    is_batch = isinstance(prompt, list)
    prompts_list = prompt if is_batch else [prompt]

    # Pre-detect multimodal token markers across the entire micro-batch
    has_multimodal = any(
        isinstance(p, str) and "<image>" in p for p in prompts_list
    )

    # Conditional loading and binary Base64 encoding from the local file system
    if has_multimodal and image_path and not image_base64:
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # 🔄 Invoke the extracted prompt mapping helper
    execution_prompts = _prepare_uqlm_execution_prompts(
        prompts_list=prompts_list, image_base64=image_base64
    )

    # Invoke the text generation and uncertainty evaluation runtime
    uqlm_result = await uqlm_wrapper.generate_and_score(prompts=execution_prompts)
    res_dict = uqlm_result.to_dict()

    responses_pool = res_dict["data"]["responses"]
    scores_pool = res_dict["data"][result_key]

    output_results = []
    for idx, current_prompt in enumerate(prompts_list):
        payload = {
            "library": "uqlm",
            "estimator_name": technique_name,
            "granularity": granularity,
            "input_prompt": current_prompt,
            "generated_text": responses_pool[idx],
            "raw_uq_result": uqlm_result,
        }

        if granularity == "sequence":
            payload["uncertainty_score"] = scores_pool[idx]
        elif granularity == "claim":
            payload["uncertainty_score"] = [
                {"claim_text": c["claim"], "score": c[result_key]}
                for c in res_dict["data"]["claims_data"][idx]
            ]
        output_results.append(payload)

    return output_results if is_batch else output_results[0]


# =====================================================================
# 2. CENTRAL ACCESS INTERFACE (Public Notebook API)
# =====================================================================

async def evaluate_uncertainty(
    prompt: Union[str, List[str]],
    technique_name: str,
    granularity: str,
    uq_context: Any,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    model_alias: str = "default",
    library: Optional[Union[str, UQLibrary]] = None,  
    **kwargs
) -> Union[dict, List[dict]]:
    """
    Unified entry point for uncertainty quantification executions.
    Performs smart lookup across libraries, handling potential name collisions.
    """
    found_lib = None
    tech_info = None

    # --- 1. Smart Lookup & Disambiguation Logic ---
    if library:
        # Normalizzazione dell'input della libreria (può essere l'Enum o la stringa es. "uqlm")
        lib_enum = library if isinstance(library, UQLibrary) else next((l for l in UQLibrary if l.value == library), None)
        if lib_enum and lib_enum in UQ_REGISTRY and technique_name in UQ_REGISTRY[lib_enum]:
            found_lib = lib_enum
            tech_info = UQ_REGISTRY[lib_enum][technique_name]
    else:
        # Ricerca automatica della tecnica in tutte le librerie registrate
        matches = []
        for lib_enum, techniques in UQ_REGISTRY.items():
            if technique_name in techniques:
                matches.append((lib_enum, techniques[technique_name]))
        
        if len(matches) == 1:
            found_lib, tech_info = matches[0]
        elif len(matches) > 1:
            # ⚠️ Gestione dell'Omonimia: Esiste lo stesso nome in più librerie!
            available_libs = [m[0].value for m in matches]
            raise ValueError(
                f"❌ Ambiguity Error: The technique '{technique_name}' is present in multiple libraries ({available_libs}). "
                f"Please specify the 'library' parameter (e.g., library=UQLibrary.POLYGRAPH or library='uqlm')."
            )

    if not tech_info or not found_lib:
        raise ValueError(f"❌ The requested technique '{technique_name}' is not registered inside UQ_REGISTRY.")

    # --- 2. Routing basato sulla libreria identificata ---
    if found_lib == UQLibrary.POLYGRAPH:
        if isinstance(prompt, list):
            raise NotImplementedError(
                "The lm_polygraph processing wrapper currently supports single-prompt executions only."
            )

        if model_alias == "default" and "default" not in uq_context.polygraph_models:
            available_models = list(uq_context.polygraph_models.keys())
            if len(available_models) == 1:
                model_alias = available_models[0]

        polygraph_model = uq_context.polygraph_models[model_alias]
        return _handle_polygraph_execution(
            prompt,
            tech_info,
            granularity,
            polygraph_model,
            image_path=image_path,
            **kwargs
        )

    elif found_lib == UQLibrary.UQLM:
        if model_alias == "default" and "default" not in uq_context.langchain_llms:
            available_models = list(uq_context.langchain_llms.keys())
            if len(available_models) == 1:
                model_alias = available_models[0]

        return await _handle_uqlm_execution(
            prompt=prompt,
            technique_name=technique_name,
            tech_info=tech_info,
            granularity=granularity,
            uq_engine=uq_context,
            model_alias=model_alias,
            image_path=image_path,
            image_base64=image_base64,
            **kwargs
        )
