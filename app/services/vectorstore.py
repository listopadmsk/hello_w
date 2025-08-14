from __future__ import annotations

import os
import json
from typing import List, Dict, Tuple

import numpy as np

try:
	import faiss  # type: ignore
except Exception:
	faiss = None  # type: ignore


class VectorStore:
	def __init__(self, storage_dir: str, dim: int):
		self.storage_dir = storage_dir
		self.dim = dim
		self.index_path = os.path.join(storage_dir, "faiss.index")
		self.meta_path = os.path.join(storage_dir, "meta.json")
		self.vecs_path = os.path.join(storage_dir, "vectors.npy")

		self._texts: List[str] = []
		self._namespaces: List[str] = []
		self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)

		self._use_faiss: bool = faiss is not None
		self._index = None
		if self._use_faiss:
			self._index = faiss.IndexFlatIP(dim)

		self._load()

	def _load(self) -> None:
		os.makedirs(self.storage_dir, exist_ok=True)
		# Load metadata
		if os.path.exists(self.meta_path):
			try:
				with open(self.meta_path, "r", encoding="utf-8") as f:
					meta = json.load(f)
				self._texts = meta.get("texts", [])
				self._namespaces = meta.get("namespaces", [])
			except Exception:
				pass
		# Load vectors
		if os.path.exists(self.vecs_path):
			try:
				mat = np.load(self.vecs_path).astype(np.float32)
				if mat.ndim == 2 and (mat.shape[1] == self.dim):
					self._matrix = mat
			except Exception:
				pass
		# Load or rebuild FAISS index
		if self._use_faiss:
			try:
				if os.path.exists(self.index_path):
					self._index = faiss.read_index(self.index_path)
			except Exception:
				self._index = faiss.IndexFlatIP(self.dim)
			if self._index is None:
				self._index = faiss.IndexFlatIP(self.dim)
			# Ensure index contains current vectors
			if self._matrix.size and (self._index.ntotal == 0 or self._index.ntotal != self._matrix.shape[0]):
				self._index.reset()
				self._index.add(self._matrix.astype(np.float32))

	def _save(self) -> None:
		with open(self.meta_path, "w", encoding="utf-8") as f:
			json.dump({"texts": self._texts, "namespaces": self._namespaces}, f, ensure_ascii=False)
		np.save(self.vecs_path, self._matrix.astype(np.float32))
		if self._use_faiss and self._index is not None:
			faiss.write_index(self._index, self.index_path)

	def add(self, embeddings: np.ndarray, texts: List[str], namespace: str = "default") -> int:
		if embeddings.size == 0:
			return 0
		if embeddings.dtype != np.float32:
			embeddings = embeddings.astype(np.float32)
		# normalize for inner product similarity
		norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
		vecs = embeddings / norms
		# append to matrix
		if self._matrix.size == 0:
			self._matrix = vecs.copy()
		else:
			self._matrix = np.vstack([self._matrix, vecs])
		# add to faiss if available
		if self._use_faiss and self._index is not None:
			self._index.add(vecs)
		# update meta
		self._texts.extend(texts)
		self._namespaces.extend([namespace] * len(texts))
		self._save()
		return len(texts)

	def search(self, query_vec: np.ndarray, top_k: int = 4, namespace: str | None = None) -> List[Tuple[int, float]]:
		if query_vec.ndim == 1:
			query_vec = query_vec.reshape(1, -1)
		norm = np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-8
		q = query_vec / norm
		if self._matrix.shape[0] == 0:
			return []
		# namespace filtering
		if namespace is not None and self._namespaces:
			indices = [i for i, ns in enumerate(self._namespaces) if ns == namespace]
			if not indices:
				return []
			mat = self._matrix[indices]
			sims = (q @ mat.T)[0]
			order = np.argsort(-sims)[:top_k]
			return [(indices[int(i)], float(sims[int(i)])) for i in order]
		# global search
		if self._use_faiss and self._index is not None:
			D, I = self._index.search(q.astype(np.float32), top_k)
			return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]
		sims = (q @ self._matrix.T)[0]
		order = np.argsort(-sims)[:top_k]
		return [(int(i), float(sims[int(i)])) for i in order]

	def get_text(self, idx: int) -> str:
		return self._texts[idx]

	def stats(self) -> Dict[str, int]:
		return {
			"num_texts": len(self._texts),
			"dim": self.dim,
		}