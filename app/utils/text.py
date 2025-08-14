from __future__ import annotations

from typing import List


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
	text = text.strip()
	if not text:
		return []
	chunks: List[str] = []
	start = 0
	while start < len(text):
		end = min(len(text), start + max_chars)
		chunks.append(text[start:end])
		start = end - overlap
		if start < 0:
			start = 0
	return chunks