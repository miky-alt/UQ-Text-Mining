import abc
from typing import Optional
from uqlm.utils import LLMGrader

class BaseResponseEvaluator(abc.ABC):
    """
    Abstract Base Class acting as a unified contract for evaluating VLM
    response quality against ground-truth answers.
    """
    @abc.abstractmethod
    async def __call__(self, prompt: str, generated_text: str, ground_truth: str) -> float:
        pass


class SubstringMatchEvaluator(BaseResponseEvaluator):
    """
    Standard deterministic evaluation strategy checking if the target
    ground-truth string is contained anywhere within the decoded text.
    """
    async def __call__(self, prompt: str, generated_text: str, ground_truth: str) -> float:
        clean_gen = generated_text.strip().lower()
        clean_truth = ground_truth.strip().lower()
        return 1.0 if clean_truth in clean_gen else 0.0
    

class LLMGraderEvaluator(BaseResponseEvaluator):
    """
    Wrapper that utilizes the existing LLMGrader class for evaluation purposes.
    Native asynchronous execution tied to the main event loop.
    """
    def __init__(self, grader: LLMGrader):
        self.grader = grader

    async def __call__(self, prompt: str, generated_text: str, ground_truth: str) -> float:
        # Esecuzione nativa asincrona senza thread pool o loop paralleli
        grades = await self.grader.grade_responses(
            prompts=[prompt],
            responses=[generated_text],
            answers=[ground_truth]
        )
        return float(grades[0])

substring_evaluator = SubstringMatchEvaluator()
