from __future__ import annotations

import os
from typing import List, Tuple

from app.services.config import AppConfig
from app.services.embeddings import Embeddings
from app.services.vectorstore import VectorStore
from app.utils.text import chunk_text
from app.utils.loaders import load_documents_from_dir


class IndexService:
	def __init__(self, config: AppConfig):
		self.config = config
		# Determine embedding dimension
		if config.embedding_provider == "hash":
			self.dim = 384
		else:
			# OpenAI text-embedding-3-small: 1536; but we detect from first call if needed
			self.dim = 1536
		self.embeddings = Embeddings(config)
		self.store = VectorStore(storage_dir=config.storage_dir, dim=self.dim)

	def stats(self) -> dict:
		return self.store.stats()

	async def ingest_directory(self, dir_path: str, namespace: str = "default") -> int:
		docs = load_documents_from_dir(dir_path)
		return await self._ingest_texts(docs, namespace=namespace)

	async def ingest_blobs(self, blobs: List[Tuple[str, bytes]], namespace: str = "default") -> int:
		docs: List[str] = []
		for name, data in blobs:
			text = data.decode("utf-8", errors="ignore")
			docs.append(text)
		return await self._ingest_texts(docs, namespace=namespace)

	async def _ingest_texts(self, docs: List[str], namespace: str) -> int:
		chunks: List[str] = []
		for d in docs:
			chunks.extend(chunk_text(d))
		if not chunks:
			return 0
		vecs = self.embeddings.embed_texts(chunks)
		return self.store.add(vecs, chunks, namespace=namespace)

	def retrieve(self, query: str, top_k: int = 4, namespace: str | None = None) -> List[tuple[str, float]]:
		qv = self.embeddings.embed_texts([query])[0]
		hits = self.store.search(qv, top_k=top_k, namespace=namespace)
		return [(self.store.get_text(i), score) for i, score in hits]