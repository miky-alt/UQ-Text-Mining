from __future__ import annotations

import gc
from typing import Any, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from lm_polygraph import CausalLMWithUncertainty
from luh import AutoUncertaintyHead
from luh.calculator_infer_luh import CalculatorInferLuh
from luh.luh_estimator_dummy import LuhEstimatorDummy


class SupervisedUQManager:
    """
    Loads a causal language model together with a pretrained supervised
    uncertainty head and exposes a generation adapter.
    """

    def __init__(
        self,
        model_name: str,
        uncertainty_head_name: str,
        *,
        device_map: Union[str, dict] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 50,
        max_input_length: int = 512,
        do_sample: bool = False,
        attn_implementation: Optional[str] = "eager",
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.uncertainty_head_name = uncertainty_head_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length
        self.do_sample = do_sample
        self.attn_implementation = attn_implementation
        self.token = token
        self.cache_dir = cache_dir
        self.model_kwargs = model_kwargs

        self.tokenizer: Optional[Any] = None
        self.base_model: Optional[Any] = None
        self.uncertainty_head: Optional[Any] = None
        self.adapter: Optional[Any] = None
        self.is_loaded = False

    @property
    def input_device(self) -> torch.device:
        if self.base_model is None:
            raise RuntimeError("The model is not loaded. Call load() first.")

        return self.base_model.get_input_embeddings().weight.device

    def load(self) -> "SupervisedUQManager":
        if self.is_loaded:
            return self

        dtype = self.torch_dtype
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer_kwargs = {}
        if self.token:
            tokenizer_kwargs["token"] = self.token
        if self.cache_dir:
            tokenizer_kwargs["cache_dir"] = self.cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither pad_token_id nor eos_token_id."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_load_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device_map if torch.cuda.is_available() else None,
            "low_cpu_mem_usage": True,
        }

        if self.attn_implementation:
            model_load_kwargs["attn_implementation"] = self.attn_implementation
        if self.token:
            model_load_kwargs["token"] = self.token
        if self.cache_dir:
            model_load_kwargs["cache_dir"] = self.cache_dir

        model_load_kwargs.update(self.model_kwargs)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_load_kwargs,
        )
        self.base_model.eval()

        head_kwargs = {}
        if self.token:
            head_kwargs["token"] = self.token
        if self.cache_dir:
            head_kwargs["cache_dir"] = self.cache_dir

        self.uncertainty_head = AutoUncertaintyHead.from_pretrained(
            self.uncertainty_head_name,
            base_model=self.base_model,
            **head_kwargs,
        )

        generation_config = GenerationConfig.from_pretrained(
            self.model_name,
            token=self.token,
            cache_dir=self.cache_dir,
        )
        generation_config.do_sample = self.do_sample
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id

        calculator = CalculatorInferLuh(
            self.uncertainty_head,
            tokenize=True,
            args_generate={
                "generation_config": generation_config,
                "max_new_tokens": self.max_new_tokens,
            },
            device=str(self.input_device),
            generations_cache_dir="",
            predict_token_uncertainties=True,
        )

        self.adapter = CausalLMWithUncertainty(
            self.base_model,
            tokenizer=self.tokenizer,
            stat_calculators=[calculator],
            estimator=LuhEstimatorDummy(),
        )

        self.is_loaded = True
        return self

    def unload(self) -> None:
        self.adapter = None
        self.uncertainty_head = None
        self.base_model = None
        self.tokenizer = None
        self.is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
