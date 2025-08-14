import os
from dataclasses import dataclass


@dataclass
class AppConfig:
	storage_dir: str
	embedding_provider: str
	llm_provider: str

	openai_api_key: str | None
	openai_embedding_model: str
	openai_chat_model: str

	ollama_base_url: str
	ollama_model: str

	@staticmethod
	def from_env() -> "AppConfig":
		storage_dir = os.getenv("STORAGE_DIR", "storage")
		embedding_provider = os.getenv("EMBEDDING_PROVIDER", "hash").lower()
		llm_provider = os.getenv("LLM_PROVIDER", "stub").lower()

		openai_api_key = os.getenv("OPENAI_API_KEY")
		openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
		openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

		ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
		ollama_model = os.getenv("OLLAMA_MODEL", "llama3:latest")

		os.makedirs(storage_dir, exist_ok=True)
		return AppConfig(
			storage_dir=storage_dir,
			embedding_provider=embedding_provider,
			llm_provider=llm_provider,
			openai_api_key=openai_api_key,
			openai_embedding_model=openai_embedding_model,
			openai_chat_model=openai_chat_model,
			ollama_base_url=ollama_base_url,
			ollama_model=ollama_model,
		)