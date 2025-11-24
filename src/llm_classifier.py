"""lazy zero-shot client for the codebase

tiny LLMClient wrapper around HF zero-shot pipelines with
lazy init, a close() method, and a module-level shared factory
"""
from __future__ import annotations
from typing import Optional, List, Dict, Union
from threading import Lock
import torch


class LLMClient:
    """Lazy zero-shot classification client.
    - Lazily initializes the HF pipeline on first use.
    - Keeps a small surface API so other modules don't need to import
      transformers directly.
    """

    def __init__(self, model: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0") -> None:
        self.model = model
        self._pipeline = None
        # protect pipeline init/close so concurrent callers don't race
        self._init_lock = Lock()

    def _init_pipeline(self):
        # double-checked locking: fast path for already-initialized pipeline
        if self._pipeline is not None:
            return
        with self._init_lock:
            if self._pipeline is not None:
                return
            try:
                # local import so module-level import doesn't require transformers
                from transformers import pipeline
                
                # Determine device: 0 for GPU, -1 for CPU
                device = 0 if torch.cuda.is_available() else -1
                print(f"Initializing pipeline on device: {device} (GPU available: {torch.cuda.is_available()})")

                self._pipeline = pipeline(
                    "zero-shot-classification", 
                    model=self.model, 
                    device=device
                )
            except Exception as e:
                # keep _pipeline as None and raise on classify; callers can catch
                self._pipeline = None
                raise RuntimeError(f"failed to initialize zero-shot pipeline: {e}")

    def classify(self, text: Union[str, List[str]], candidate_labels: List[str], hypothesis_template: str = "This is {}.", batch_size: int = 32) -> Union[Dict, List[Dict]]:
        # Run zero-shot classification and return the pipeline output.
        if self._pipeline is None:
            self._init_pipeline()

        if self._pipeline is None:
            raise RuntimeError("zero-shot pipeline not available")

        result = self._pipeline(text, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template, batch_size=batch_size)
        return result

    def close(self) -> None:
        # protect against concurrent init/close races
        with self._init_lock:
            if self._pipeline is None:
                return
            # attempt to move model to CPU (if present) before dropping
            try:
                if hasattr(self._pipeline, "model"):
                    try:
                        self._pipeline.model.cpu()
                    except Exception:
                        pass
            except Exception:
                pass
            # drop the pipeline reference
            try:
                self._pipeline = None
            except Exception:
                self._pipeline = None

        # best-effort free CUDA memory
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass


# convenience factory for a module-level shared client
_shared_client: Optional[LLMClient] = None


def get_shared_zero_shot(model: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0") -> LLMClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = LLMClient(model=model)
    return _shared_client


def reset_shared_zero_shot() -> None:
    """Reset and close the module-level shared LLM client (if any)."""
    global _shared_client
    if _shared_client is not None:
        try:
            _shared_client.close()
        except Exception:
            pass
    _shared_client = None
