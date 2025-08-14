import os
import asyncio
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.services.config import AppConfig
from app.services.index import IndexService
from app.services.rag import RAGService

load_dotenv()

app = FastAPI(title="RAG Service", version="0.1.0")

config = AppConfig.from_env()
index_service = IndexService(config=config)
rag_service = RAGService(config=config, index_service=index_service)


class QueryRequest(BaseModel):
	query: str
	top_k: int = 4


@app.get("/health")
async def health() -> JSONResponse:
	return JSONResponse({"status": "ok"})


@app.get("/stats")
async def stats() -> JSONResponse:
	return JSONResponse(index_service.stats())


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(default=[]),
			  namespace: Optional[str] = Form(default="default")) -> JSONResponse:
	"""Ingest uploaded files into the index under a namespace."""
	contents: List[tuple[str, bytes]] = []
	for f in files:
		contents.append((f.filename, await f.read()))
	count = await index_service.ingest_blobs(blobs=contents, namespace=namespace)
	return JSONResponse({"ingested": count, "namespace": namespace})


@app.post("/query")
async def query(req: QueryRequest) -> JSONResponse:
	results = await rag_service.query(req.query, top_k=req.top_k)
	return JSONResponse(results)