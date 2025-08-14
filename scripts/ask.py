import asyncio
import sys
from dotenv import load_dotenv

from app.services.config import AppConfig
from app.services.index import IndexService
from app.services.rag import RAGService


async def main() -> int:
	load_dotenv()
	if len(sys.argv) < 2:
		print("Usage: python scripts/ask.py <question> [top_k]")
		return 2
	question = sys.argv[1]
	top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
	config = AppConfig.from_env()
	index = IndexService(config)
	rag = RAGService(config, index)
	res = await rag.query(question, top_k=top_k)
	print(res["answer"]) 
	return 0


if __name__ == "__main__":
	sys.exit(asyncio.run(main()))