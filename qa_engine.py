"""Small, transparent retrieval-based QA engine with conservative abstention.

No generative model or outside knowledge is used. It retrieves source chunks and
returns an answer only when a chunk passes a similarity threshold and contains
a likely answer-bearing passage. This is a teaching/demo implementation, not a
production semantic search system.
"""
from pathlib import Path
import re
import math
from collections import Counter

DEFAULT_THRESHOLD = 0.18

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class ContextQA:
    def __init__(self, chunks, source_names, threshold=DEFAULT_THRESHOLD):
        self.chunks = chunks
        self.source_names = source_names
        self.threshold = threshold
        self.idf = {}
        self.vectors = []
        self._fit()

    @classmethod
    def from_directory(cls, directory, threshold=DEFAULT_THRESHOLD):
        root = Path(directory)
        chunks = []
        names = []
        for path in sorted(root.glob("*.md")) + sorted(root.glob("*.txt")):
            names.append(path.name)
            text = path.read_text(encoding="utf-8")
            section = "General"
            for block in re.split(r"\n(?=##?\s)", text):
                lines = block.strip().splitlines()
                if not lines:
                    continue
                heading = lines[0].lstrip("# ").strip() if lines[0].startswith("#") else section
                body = "\n".join(lines[1:]).strip() if lines[0].startswith("#") else block.strip()
                if body:
                    chunks.append({"file": path.name, "section": heading or "General", "text": body})
                section = heading or section
        return cls(chunks, names, threshold)

    def _fit(self):
        docs = [Counter(set(tokenize(c["text"]))) for c in self.chunks]
        n = max(1, len(docs))
        vocab = set(t for d in docs for t in d)
        for term in vocab:
            df = sum(1 for d in docs if term in d)
            self.idf[term] = math.log(1 + (n + 1) / (df + 1)) + 1
        self.vectors = [self._vector(tokenize(c["text"])) for c in self.chunks]

    def _vector(self, tokens):
        counts = Counter(tokens)
        vec = {t: (1 + math.log(cnt)) * self.idf.get(t, 0.0) for t, cnt in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _similarity(self, query, vector):
        qv = self._vector(tokenize(query))
        return sum(value * vector.get(term, 0.0) for term, value in qv.items())

    def retrieve(self, question, top_k=3):
        scored = [(self._similarity(question, vec), chunk) for vec, chunk in zip(self.vectors, self.chunks)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(score, chunk) for score, chunk in scored[:top_k]]

    def answer(self, question):
        question = question.strip()
        if not question:
            return {"answered": False, "answer": "Please enter a question.", "sources": []}
        retrieved = self.retrieve(question)
        if not retrieved or retrieved[0][0] < self.threshold:
            return {
                "answered": False,
                "answer": "I can’t answer that from the provided knowledge base. The source does not contain enough relevant information.",
                "sources": []
            }
        best_score, best = retrieved[0]
        # Return the most relevant source passage verbatim to avoid unsupported synthesis.
        # A passage is shown as evidence; the UI labels it clearly as source-grounded.
        sources = []
        for score, chunk in retrieved[:2]:
            if score >= max(self.threshold, best_score * 0.55):
                sources.append({
                    "file": chunk["file"], "section": chunk["section"],
                    "text": chunk["text"], "score": score
                })
        if not sources:
            return {"answered": False, "answer": "I can’t answer that from the provided knowledge base. The source does not contain enough relevant information.", "sources": []}
        answer = f"Based on the knowledge base: {best['text']}"
        return {"answered": True, "answer": answer, "sources": sources}
