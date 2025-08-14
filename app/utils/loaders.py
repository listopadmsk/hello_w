from __future__ import annotations

import os
from typing import List


def load_documents_from_dir(dir_path: str) -> List[str]:
	texts: List[str] = []
	if not os.path.isdir(dir_path):
		return texts
	for root, _, files in os.walk(dir_path):
		for name in files:
			path = os.path.join(root, name)
			low = name.lower()
			if low.endswith((".txt", ".md")):
				try:
					with open(path, "r", encoding="utf-8") as f:
						texts.append(f.read())
				except Exception:
					continue
			elif low.endswith(".pdf"):
				try:
					from pypdf import PdfReader
				except Exception:
					continue
				try:
					reader = PdfReader(path)
					t = "\n\n".join(page.extract_text() or "" for page in reader.pages)
					if t.strip():
						texts.append(t)
				except Exception:
					continue
	return texts