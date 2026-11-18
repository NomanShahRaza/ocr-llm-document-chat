from __future__ import annotations

import threading
from typing import Sequence

import numpy as np

from ..config import BGE_M3_MODEL, RAG_DEVICE


class BgeM3Encoder:
    """Dense encoder for BAAI/bge-m3 using transformers (already in the env)."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or BGE_M3_MODEL
        self.device = (device or RAG_DEVICE or 'cpu').lower()
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = self.device
            if device.startswith('cuda') and not torch.cuda.is_available():
                device = 'cpu'
            self.device = device
            # float16 cuts RAM roughly in half; important on 16GB hosts with OCR loaded.
            dtype = torch.float16
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            load_kwargs = {'low_cpu_mem_usage': True}
            try:
                self._model = AutoModel.from_pretrained(self.model_name, dtype=dtype, **load_kwargs)
            except TypeError:
                self._model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype, **load_kwargs)
            self._model.to(device)
            self._model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 4) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        self.ensure_loaded()
        import torch

        vectors: list[np.ndarray] = []
        with self._lock:
            assert self._tokenizer is not None and self._model is not None
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    # Chunks are ~900 chars; 512 tokens is enough and much faster on CPU.
                    max_length=512,
                    return_tensors='pt',
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    output = self._model(**inputs)
                    # BGE dense retrieval uses CLS pooling + L2 normalize.
                    emb = output.last_hidden_state[:, 0]
                    emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
                vectors.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.vstack(vectors)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None


encoder = BgeM3Encoder()
