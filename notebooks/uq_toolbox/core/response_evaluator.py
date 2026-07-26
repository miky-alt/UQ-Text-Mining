import abc
class BaseResponseEvaluator(abc.ABC):
    """
    Abstract Base Class acting as a unified contract for evaluating VLM
    response quality against ground-truth clinical answers.
    """
    @abc.abstractmethod
    async def __call__(self, prompt: str, generated_text: str, ground_truth: str) -> float:
        pass


class SubstringMatchEvaluator(BaseResponseEvaluator):
    """
    Standard deterministic evaluation strategy checking if the target
    ground-truth string is contained anywhere within the decoded text.
    """
    def __call__(self, generated_text: str, ground_truth: str) -> float:
        clean_gen = generated_text.strip().lower()
        clean_truth = ground_truth.strip().lower()
        return 1.0 if clean_truth in clean_gen else 0.0
    
class LLMGraderEvaluator(BaseResponseEvaluator):
    """
    Wrapper that utilizes the existing LLMGrader class for evaluation purposes.
    """
    def __init__(self, grader):
        self.grader = grader

    async def __call__(self, prompt: str, generated_text: str, ground_truth: str) -> float:
        # LLMGrader.grade_responses expects lists; wrap individual inputs as single-element lists
        grades = await self.grader.grade_responses(
            prompts=[prompt],
            responses=[generated_text],
            answers=[ground_truth]
        )
        # Return the grade as a float (e.g., 1.0 or 0.0)
        return float(grades[0])