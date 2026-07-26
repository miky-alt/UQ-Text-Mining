# __init__.py

# 1. Versioning (Pratica standard per librerie professionali)
__version__ = "0.1.0"

# 2. Main API (Le tue funzioni principali)
from .core.pipeline import compute_dataset_uq_scores, compute_batch_uqlm_scores
from .core.uq_engine import evaluate_uncertainty
from .registry import UQ_REGISTRY

# 3. Evaluators (Esponi le classi per l'iniezione delle dipendenze)
from .core.response_evaluator import BaseResponseEvaluator, SubstringMatchEvaluator

# 4. Managers (Se necessario, esponi i manager)
from .managers.llama_cpp_manager import LlamaCppManager

# 5. Learned UQ (Backend con supervised uncertainty head)
# Optional: requires the separate `luh` (llm-uncertainty-head) package. Falls back
# gracefully so the rest of the toolbox (registry/engine/managers) still works without it.
try:
    from .learned_uq import (
        SupervisedUQManager,
        evaluate_supervised_uncertainty,
        evaluate_supervised_batch,
    )
    _HAS_LEARNED_UQ = True
except ImportError:
    _HAS_LEARNED_UQ = False

# 6. Public API Control (La lista bianca delle API esportate)
__all__ = [
    "compute_dataset_uq_scores",
    "compute_batch_uqlm_scores",
    "evaluate_uncertainty",
    "UQ_REGISTRY",
    "BaseResponseEvaluator",
    "SubstringMatchEvaluator",
    "LlamaCppManager",
    "__version__",
]

if _HAS_LEARNED_UQ:
    __all__ += [
        "SupervisedUQManager",
        "evaluate_supervised_uncertainty",
        "evaluate_supervised_batch",
    ]
