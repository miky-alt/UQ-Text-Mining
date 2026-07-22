from typing import Any, Dict, List, Optional

from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.estimators import RDESeq
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.factory_stat_calculator import (
    StatCalculatorContainer,
)
from lm_polygraph.utils.manager import UEManager
from omegaconf import OmegaConf


def _build_rde_training_config(
    training_size: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Build the dataset configuration required for RDE training statistics."""
    return {
        "dataset": ["qiaojin/PubMedQA", "pqa_labeled"],
        "train_dataset": ["qiaojin/PubMedQA", "pqa_labeled"],
        "text_column": "question",
        "label_column": "final_decision",
        "train_split": "train",
        "few_shot_split": "train",
        "prompt": "{question}",

        "size": training_size,
        "subsample_train_dataset": training_size,
        "batch_size": batch_size,
        "seed": seed,

        # Required by LM-Polygraph
        "description": "",
        "load_from_disk": False,
        "train_test_split": False,
        "n_shot": 0,
        "bg_size": 1,

        "background_train_dataset": "allenai/c4",
        "background_train_dataset_text_column": "text",
        "background_train_dataset_label_column": "url",
        "background_train_dataset_data_files": (
            "en/c4-train.00000-of-01024.json.gz"
        ),
        "background_load_from_disk": False,
        "subsample_background_train_dataset": 1,
    }


def run_rde_sequence(
    uq_context: Any,
    prompts: List[str],
    model_alias: str = "qwen",
    training_size: int = 20,
    batch_size: int = 1,
    max_new_tokens: int = 8,
    seed: int = 42,
    layer: str = "decoder",
    training_config_overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Run LM-Polygraph Robust Density Estimation for sequence-level UQ.

    Args:
        uq_context:
            Toolbox model manager containing registered Polygraph models.
        prompts:
            Prompts to evaluate.
        model_alias:
            Alias used in uq_context.polygraph_models.
        training_size:
            Number of PubMedQA examples used to estimate reference statistics.
        batch_size:
            Evaluation and training-statistics batch size.
        max_new_tokens:
            Maximum number of generated tokens.
        seed:
            Dataset sampling seed.
        layer:
            RDE representation layer, normally "decoder".
        training_config_overrides:
            Optional values that override the default training configuration.

    Returns:
        The result returned by LM-Polygraph's UEManager.
    """
    if not prompts:
        raise ValueError("prompts must contain at least one prompt.")

    if model_alias not in uq_context.polygraph_models:
        available = list(uq_context.polygraph_models)

        raise KeyError(
            f"No Polygraph model registered as '{model_alias}'. "
            f"Available aliases: {available}"
        )

    if training_size <= 0:
        raise ValueError("training_size must be greater than zero.")

    polygraph_model = uq_context.polygraph_models[model_alias]

    evaluation_dataset = Dataset(
        prompts,
        [""] * len(prompts),
        batch_size=batch_size,
    )

    training_config = _build_rde_training_config(
        training_size=training_size,
        batch_size=batch_size,
        seed=seed,
    )

    if training_config_overrides:
        training_config.update(training_config_overrides)

    calculators = {
        calculator.name: calculator
        for calculator in register_default_stat_calculators("Whitebox")
    }

    calculators["TrainingStatisticExtractionCalculator"] = (
        StatCalculatorContainer(
            name="TrainingStatisticExtractionCalculator",
            cfg=OmegaConf.create(training_config),
            stats=["train_embeddings"],
            dependencies=[],
            builder=(
                "lm_polygraph.defaults.stat_calculator_builders."
                "default_TrainingStatisticExtractionCalculator"
            ),
        )
    )

    manager = UEManager(
        data=evaluation_dataset,
        model=polygraph_model,
        estimators=[RDESeq(layer)],
        builder_env_stat_calc=BuilderEnvironmentStatCalculator(
            model=polygraph_model,
        ),
        available_stat_calculators=list(calculators.values()),
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        max_new_tokens=max_new_tokens,
    )

    print(
        f"Running RDE with {training_size} reference examples "
        f"for {len(prompts)} prompt(s)..."
    )

    result = manager()

    print("RDE calculation complete.")
    return result
