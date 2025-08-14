from __future__ import annotations

from typing import List, Dict

from app.services.config import AppConfig
from app.services.index import IndexService
from app.services.llm import LLM


RAG_SYSTEM_PROMPT = (
	"You are a helpful assistant. Use the provided CONTEXT to answer the question. "
	"If the answer cannot be found in the context, say you do not know."
)


class RAGService:
	def __init__(self, config: AppConfig, index_service: IndexService):
		self.config = config
		self.index = index_service
		self.llm = LLM(config)

	async def query(self, question: str, top_k: int = 4) -> Dict[str, object]:
		retrieved = self.index.retrieve(question, top_k=top_k)
		context_texts = [t for t, _ in retrieved]
		context = "\n\n---\n\n".join(context_texts)
		user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in Russian."
		answer = await self.llm.generate(system_prompt=RAG_SYSTEM_PROMPT, user_prompt=user_prompt)
		return {
			"answer": answer,
			"sources": [
				{"text": t, "score": s}
				for t, s in retrieved
			],
		}