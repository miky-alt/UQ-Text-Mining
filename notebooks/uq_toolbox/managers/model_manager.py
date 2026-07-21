# --- Standard Library ---
import os
import getpass
import gc
import torch
from typing import Optional, Union, Any, Dict, List

# --- Third-Party / ML Libraries ---
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor, 
    AutoModelForImageTextToText, 
    BitsAndBytesConfig
)

# --- lm_polygraph Framework ---
from lm_polygraph.utils.model import BlackboxModel, WhiteboxModel
from lm_polygraph.model_adapters import VisualWhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters

# --- LangChain ---
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# --- Local Imports ---
from .llama_cpp_manager import LlamaCppManager

class UQModelManager:
    """
    Manager class for Uncertainty Quantification Models.
    Handles multi-model setups with independent operational modes per alias.
    """
    def __init__(self, default_polygraph_mode: str = "black", default_uqlm_mode: str = "black"):
        # Set default operational modes for both frameworks
        self.default_polygraph_mode = default_polygraph_mode.strip().lower()
        self.default_uqlm_mode = default_uqlm_mode.strip().lower()

        # Registries to hold instantiated models
        self.polygraph_models: dict[str, Any] = {}
        self.langchain_llms: dict[str, Any] = {}

        # Mappings for operational modes (white-box vs black-box)
        self.polygraph_modes: dict[str, str] = {}
        self.uqlm_modes: dict[str, str] = {}

    # ==========================================
    # REGISTRY MANAGEMENT
    # ==========================================
    def list_active_models(self) -> None:
        """
        Prints a structural summary of all currently live model aliases
        and their assigned operational boundaries.
        """
        print("\n" + "="*55)
        print("📋 LIVE UNCERTAINTY QUANTIFICATION REGISTRIES")
        print("="*55)

        poly_aliases = list(self.polygraph_models.keys())
        uqlm_aliases = list(self.langchain_llms.keys())

        print(f"🔹 lm_polygraph Framework ({len(poly_aliases)} active):")
        if not poly_aliases:
            print("  (No active aliases allocated)")
        for alias in poly_aliases:
            mode = self.polygraph_modes.get(alias, "unknown").upper()
            print(f"  • Alias: [{alias.upper()}] ➔ Operational Mode: {mode}-BOX")

        print(f"\n🔸 UQLM Calibration Framework ({len(uqlm_aliases)} active):")
        if not uqlm_aliases:
            print("  (No active aliases allocated)")
        for alias in uqlm_aliases:
            mode = self.uqlm_modes.get(alias, "unknown").upper()
            print(f"  • Alias: [{alias.upper()}] ➔ Operational Mode: {mode}-BOX")
        print("="*55 + "\n")

    # ==========================================
    # MODEL LOADING AND CONFIGURATION
    # ==========================================
    def add_polygraph_model(self,
                            model_id: str,
                            alias: str = "default",
                            mode: Optional[str] = None,
                            is_multimodal: bool = False,
                            quant_mode: str = "none",
                            temperature: float = 0.7,
                            max_new_tokens: int = 30,
                            do_sample: bool = True,
                            device_map: str = "auto",
                            attn_implementation: str = "eager",
                            **hf_extra_kwargs):
        """Adds an lm_polygraph model to the registry with independent box mode."""
        # Determine operational mode with fallback
        current_mode = mode.strip().lower() if mode else self.default_polygraph_mode
        if current_mode == "none":
            print(f"⚠️ Polygraph mode is set to 'none'. Skipping load for alias '{alias}'.")
            return

        print(f"\n⚙️ Adding lm_polygraph [{alias.upper()}] in {current_mode.upper()}-BOX mode...")
        self.polygraph_modes[alias] = current_mode

        shared_generation_params = GenerationParameters(
            temperature=temperature, do_sample=do_sample, max_new_tokens=max_new_tokens
        )

        # Handle local white-box model loading
        if current_mode == "white":
            if "HF_TOKEN" not in os.environ:
                os.environ["HF_TOKEN"] = getpass.getpass("Paste your Hugging Face Token: ")

            model_loader_class = hf_extra_kwargs.pop("model_loader_class", AutoModelForImageTextToText)
            model_kwargs = {"device_map": device_map if torch.cuda.is_available() else "cpu", "token": os.environ["HF_TOKEN"]}

            if attn_implementation:
                model_kwargs["attn_implementation"] = attn_implementation

            if "trust_remote_code" not in hf_extra_kwargs:
                model_kwargs["trust_remote_code"] = True

            model_kwargs.update(hf_extra_kwargs)

            # Apply quantization config if requested
            if quant_mode == '4bit':
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            elif quant_mode == '8bit':
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                model_kwargs["torch_dtype"] = torch.float16

            # Instantiate model based on multimodal flag
            if is_multimodal:
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    token=os.environ["HF_TOKEN"],
                    trust_remote_code=model_kwargs.get("trust_remote_code", True)
                )
                base_model = model_loader_class.from_pretrained(model_id, **model_kwargs)
                self.polygraph_models[alias] = VisualWhiteboxModel(
                    model=base_model, processor_visual=processor, model_path=model_id, generation_parameters=shared_generation_params
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
                base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                self.polygraph_models[alias] = WhiteboxModel(
                    model=base_model, tokenizer=tokenizer, model_path=model_id, generation_parameters=shared_generation_params
                )
        # Handle API-based black-box model loading
        else:
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI API Key: ")
            if "HF_TOKEN" not in os.environ:
                os.environ["HF_TOKEN"] = getpass.getpass("Paste your Hugging Face Token: ")

            self.polygraph_models[alias] = BlackboxModel(
                model_path=model_id, openai_api_key=os.environ.get("OPENAI_API_KEY"), hf_api_token=os.environ.get("HF_TOKEN"), supports_logprobs=True, generation_parameters=shared_generation_params
            )

    def add_uqlm_model(self,
                       provider: str,
                       model_id: Optional[str] = None,
                       alias: str = "default",
                       mode: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       custom_llm: Optional[Any] = None,
                       ollama_url: str = "http://localhost:11434",
                       auto_install: bool = False,
                       **uqlm_kwargs):
        """Adds a LangChain LLM to the registry with independent box mode mapping."""
        current_mode = mode.strip().lower() if mode else self.default_uqlm_mode
        if current_mode == "none":
            return

        provider = provider.strip().lower()

        # Enforce black-box mode for providers without white-box logprobs support
        if current_mode == "white" and provider in ["anthropic", "google"]:
            print(f"⚠️ CRITICAL WARNING: Provider '{provider.upper()}' for alias '{alias}' does not support White-Box logprobs. Forcing Black-Box.")
            current_mode = "black"

        self.uqlm_modes[alias] = current_mode
        print(f"\n📊 Adding UQLM [{alias.upper()}] in {current_mode.upper()}-BOX mode...")

        if provider == "custom":
            self.langchain_llms[alias] = custom_llm
            return

        # Handle OpenAI Provider
        if provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI Key: ")
            openai_kwargs = {"logprobs": True} if current_mode == "white" else {}
            self.langchain_llms[alias] = ChatOpenAI(model=model_id, temperature=temperature, max_tokens=max_tokens, model_kwargs=openai_kwargs, **uqlm_kwargs)
        
        # Handle Google Provider
        elif provider == "google":
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = getpass.getpass("Paste your Google Key: ")
            self.langchain_llms[alias] = ChatGoogleGenerativeAI(model=model_id, temperature=temperature, max_tokens=max_tokens, **uqlm_kwargs)
        
        # Handle Anthropic Provider
        elif provider == "anthropic":
            if "ANTHROPIC_API_KEY" not in os.environ:
                os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Paste your Anthropic Key: ")
            self.langchain_llms[alias] = ChatAnthropic(model=model_id, temperature=temperature, max_tokens=max_tokens, **uqlm_kwargs)

        # Handle LlamaCpp/Local Provider (Self-healing orchestration)
        elif provider == "llamacpp":
            print(f"\n📊 Initializing LlamaCpp for [{alias.upper()}]...")
            
            # Use LlamaCppManager for binary installation and service lifecycle
            manager = LlamaCppManager(base_url=ollama_url)

            # Ensure model is ready; returns False if download or start fails
            if not manager.load_model(model_name=model_id):
                print(f"❌ Aborting allocation for alias [{alias.upper()}]: Service failed.")
                return

            # Map to ChatOpenAI interface for compatibility with logprobs (White-box usage)
            model_kwargs = {"logprobs": True} if current_mode == "white" else {}
            
            self.langchain_llms[alias] = ChatOpenAI(
                base_url=f"{manager.base_url}/v1",
                api_key="llama-cpp-local",
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
                **uqlm_kwargs
            )
            print(f"🔗 LlamaCpp instance active at {manager.base_url}/v1")

    def remove_models(self, aliases: Optional[Union[list[str], str]] = None) -> None:
        """
        Safely removes specified model aliases or all models from memory,
        breaking references to force garbage collection and flush GPU VRAM cache.
        """
        # Determine which models to remove
        if aliases is None:
            all_targets = list(set(list(self.polygraph_models.keys()) + list(self.langchain_llms.keys())))
        elif isinstance(aliases, str):
            all_targets = [aliases.strip()]
        else:
            all_targets = [a.strip() for a in aliases]

        if not all_targets:
            print("🧹 No active models found in registries to remove.")
            return

        print(f"\n🧹 Initiating memory purge for target aliases: {all_targets}")

        for alias in all_targets:
            removed_any = False

            # Clear Polygraph-specific references
            if alias in self.polygraph_models:
                wrapper = self.polygraph_models[alias]

                if isinstance(wrapper, VisualWhiteboxModel):
                    del wrapper.model
                    del wrapper.processor_visual
                elif isinstance(wrapper, WhiteboxModel):
                    del wrapper.model
                    del wrapper.tokenizer

                del self.polygraph_models[alias]
                self.polygraph_modes.pop(alias, None)
                removed_any = True

            # Clear LangChain/UQLM references
            if alias in self.langchain_llms:
                del self.langchain_llms[alias]
                self.uqlm_modes.pop(alias, None)
                removed_any = True

            if removed_any:
                print(f"  ✅ References cleared for alias: [{alias.upper()}]")
            else:
                print(f"  ⚠️ Target alias [{alias.upper()}] not found in any active registry.")

        # Trigger garbage collection
        gc.collect()

        # Clear GPU VRAM if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  ✨ GPU VRAM cache cleared and synchronized.")
        else:
            print("  ✨ System RAM defragmented via garbage collection.")

def initialize_uq_models(
    polygraph_models: Optional[Union[str, dict]] = None,
    polygraph_is_multimodal: bool = False,
    uqlm_models: Optional[dict] = None,
    p_mode: Optional[str] = None,
    u_mode: Optional[str] = None,
    quant_choice: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    ollama_url: str = "http://localhost:11434",
    auto_install: bool = False,
    polygraph_kwargs: Optional[dict] = None,
    uqlm_kwargs: Optional[dict] = None
) -> UQModelManager:
    """
    Wrapper function to initialize and bulk-load a multi-framework UQ environment
    containing Polygraph and UQLM models.
    """
    print("\n" + "="*60)
    print("🚀 INITIALIZING MULTI-MODE UQ EXPERIMENTAL ENVIRONMENT")
    print("="*60)

    polygraph_kwargs = polygraph_kwargs or {}
    uqlm_kwargs = uqlm_kwargs or {}

    # Standardize input for Polygraph model registry
    if isinstance(polygraph_models, str):
        poly_dict = {"default": {"model_id": polygraph_models}}
    elif isinstance(polygraph_models, dict):
        poly_dict = {alias: {"model_id": val} if isinstance(val, str) else val for alias, val in polygraph_models.items()}
    else:
        poly_dict = {}

    uqlm_dict = uqlm_models or {}

    p_mode = p_mode or "black"
    u_mode = u_mode or "black"

    uq_engine = UQModelManager(default_polygraph_mode=p_mode, default_uqlm_mode=u_mode)

    # --- Process and Bulk Load Polygraph Models ---
    for alias, config in poly_dict.items():
        spec_kwargs = config.get("kwargs", {}).copy()
        reserved_keys = {"model_id", "mode", "quant_mode", "kwargs"}
        extra_flat_kwargs = {k: v for k, v in config.items() if k not in reserved_keys}
        merged_kwargs = {**polygraph_kwargs, **extra_flat_kwargs, **spec_kwargs}

        uq_engine.add_polygraph_model(
            model_id=config["model_id"],
            alias=alias,
            mode=config.get("mode", p_mode),
            quant_mode=config.get("quant_mode", quant_choice or "none"),
            temperature=temperature,
            max_new_tokens=max_tokens,
            is_multimodal=polygraph_is_multimodal,
            **merged_kwargs
        )

    # --- Process and Bulk Load UQLM Models ---
    for alias, config in uqlm_dict.items():
        spec_kwargs = config.get("kwargs", {})
        merged_kwargs = {**uqlm_kwargs, **spec_kwargs}

        uq_engine.add_uqlm_model(
            provider=config["provider"],
            model_id=config["model_id"],
            alias=alias,
            mode=config.get("mode", u_mode),
            temperature=config.get("temperature", temperature),
            max_tokens=config.get("max_tokens", max_tokens),
            ollama_url=ollama_url,
            auto_install=config.get("auto_install", auto_install),
            **merged_kwargs
        )

    print("\n" + "="*60)
    print("✅ ENVIRONMENT READY FOR QUANTIFICATION PIPELINES")
    print("="*60)
    return uq_engine