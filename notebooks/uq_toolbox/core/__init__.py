# Importa le funzioni chiave dai file interni del sottomodulo core
from .uq_engine import evaluate_uncertainty
from .pipeline import compute_dataset_uq_scores, compute_batch_uqlm_scores
from .response_evaluator import BaseResponseEvaluator, SubstringMatchEvaluator

# Definisci esplicitamente cosa viene esportato con "from uq_toolbox.core import *"
__all__ = [
    "evaluate_uncertainty",
    "compute_dataset_uq_scores",
    "compute_batch_uqlm_scores",
    "BaseResponseEvaluator",
    "SubstringMatchEvaluator",
]