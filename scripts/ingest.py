import asyncio
import sys
from dotenv import load_dotenv

from app.services.config import AppConfig
from app.services.index import IndexService


async def main() -> int:
	load_dotenv()
	if len(sys.argv) < 2:
		print("Usage: python scripts/ingest.py <directory> [namespace]")
		return 2
	dir_path = sys.argv[1]
	namespace = sys.argv[2] if len(sys.argv) > 2 else "default"
	config = AppConfig.from_env()
	index = IndexService(config)
	count = await index.ingest_directory(dir_path, namespace=namespace)
	print(f"Ingested {count} chunks into namespace '{namespace}'.")
	return 0


if __name__ == "__main__":
	sys.exit(asyncio.run(main()))