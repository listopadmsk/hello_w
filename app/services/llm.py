from __future__ import annotations

from typing import List, Dict

import httpx

try:
	from openai import OpenAI
except Exception:
	OpenAI = None  # type: ignore

from app.services.config import AppConfig


class LLM:
	def __init__(self, config: AppConfig):
		self.config = config
		self.client = None
		if config.llm_provider == "openai":
			if OpenAI is None:
				raise RuntimeError("openai package not installed")
			if not config.openai_api_key:
				raise RuntimeError("OPENAI_API_KEY missing for openai chat")
			self.client = OpenAI(api_key=config.openai_api_key)

	async def generate(self, system_prompt: str, user_prompt: str) -> str:
		provider = self.config.llm_provider
		if provider == "stub":
			return f"[STUB] {user_prompt}"
		elif provider == "openai":
			return await self._chat_openai(system_prompt, user_prompt)
		elif provider == "ollama":
			return await self._chat_ollama(system_prompt, user_prompt)
		else:
			raise ValueError(f"Unsupported LLM provider: {provider}")

	async def _chat_openai(self, system_prompt: str, user_prompt: str) -> str:
		assert self.client is not None
		resp = self.client.chat.completions.create(
			model=self.config.openai_chat_model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		)
		return resp.choices[0].message.content or ""

	async def _chat_ollama(self, system_prompt: str, user_prompt: str) -> str:
		url = f"{self.config.ollama_base_url}/api/chat"
		payload: Dict[str, object] = {
			"model": self.config.ollama_model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		}
		async with httpx.AsyncClient(timeout=60) as client:
			resp = await client.post(url, json=payload)
			resp.raise_for_status()
			data = resp.json()
			if "message" in data and isinstance(data["message"], dict):
				return data["message"].get("content", "")
			return data.get("response", "")