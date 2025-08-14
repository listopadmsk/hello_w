from __future__ import annotations

import hashlib
from typing import List

import numpy as np

try:
	from openai import OpenAI
except Exception:
	OpenAI = None  # type: ignore

from app.services.config import AppConfig


class Embeddings:
	def __init__(self, config: AppConfig):
		self.config = config
		self.client = None
		if config.embedding_provider == "openai":
			if OpenAI is None:
				raise RuntimeError("openai package not installed")
			if not config.openai_api_key:
				raise RuntimeError("OPENAI_API_KEY missing for openai embeddings")
			self.client = OpenAI(api_key=config.openai_api_key)

	def embed_texts(self, texts: List[str]) -> np.ndarray:
		provider = self.config.embedding_provider
		if provider == "hash":
			return self._hash_embeddings(texts)
		elif provider == "openai":
			return self._openai_embeddings(texts)
		else:
			raise ValueError(f"Unsupported embedding provider: {provider}")

	def _hash_embeddings(self, texts: List[str]) -> np.ndarray:
		# Deterministic 384-dim embedding via SHA256 buckets
		dim = 384
		vecs = []
		for t in texts:
			h = hashlib.sha256(t.encode("utf-8")).digest()
			# repeat hash to fill dim
			raw = (h * ((dim // len(h)) + 1))[:dim]
			arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
			arr = arr / (np.linalg.norm(arr) + 1e-8)
			vecs.append(arr)
		return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)

	def _openai_embeddings(self, texts: List[str]) -> np.ndarray:
		assert self.client is not None
		model = self.config.openai_embedding_model
		resp = self.client.embeddings.create(model=model, input=texts)
		vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
		return np.vstack(vecs)