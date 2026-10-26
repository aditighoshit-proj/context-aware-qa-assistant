"""
Context-Aware Q&A Assistant

Features:
- Retrieves relevant information from local knowledge-base files.
- Extracts the most relevant sentence as the answer.
- Returns answers based only on the supplied knowledge base.
- Abstains when sufficient supporting information is unavailable.
- Includes source information for transparency.
"""

from pathlib import Path
import re
import math
from collections import Counter


DEFAULT_THRESHOLD = 0.18

ABSTENTION_MESSAGE = (
    "I can't answer that from the provided knowledge base. "
    "The source does not contain enough relevant information."
)


def tokenize(text):
    """Convert text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class ContextQA:

    def __init__(
        self,
        chunks,
        source_names,
        threshold=DEFAULT_THRESHOLD
    ):
        self.chunks = chunks
        self.source_names = source_names
        self.threshold = threshold

        self.idf = {}
        self.vectors = []

        self._fit()

    @classmethod
    def from_directory(
        cls,
        directory,
        threshold=DEFAULT_THRESHOLD
    ):
        """
        Load Markdown and text files from the knowledge-base directory.
        Each section becomes a searchable document chunk.
        """
        root = Path(directory)

        chunks = []
        source_names = []

        files = (
            sorted(root.glob("*.md"))
            + sorted(root.glob("*.txt"))
        )

        for file_path in files:
            source_names.append(file_path.name)

            content = file_path.read_text(encoding="utf-8")

            section = "General"

            blocks = re.split(
                r"\n(?=##?\s)",
                content
            )

            for block in blocks:
                lines = block.strip().splitlines()

                if not lines:
                    continue

                if lines[0].startswith("#"):
                    section = (
                        lines[0]
                        .lstrip("# ")
                        .strip()
                        or section
                    )

                    body = "\n".join(lines[1:]).strip()

                else:
                    body = block.strip()

                if body:
                    chunks.append({
                        "file": file_path.name,
                        "section": section,
                        "text": body
                    })

        return cls(
            chunks=chunks,
            source_names=source_names,
            threshold=threshold
        )

    def _fit(self):
        """Build TF-IDF-style document vectors."""

        documents = [
            set(tokenize(chunk["text"]))
            for chunk in self.chunks
        ]

        document_count = max(1, len(documents))

        vocabulary = set(
            term
            for document in documents
            for term in document
        )

        for term in vocabulary:
            document_frequency = sum(
                term in document
                for document in documents
            )

            self.idf[term] = (
                math.log(
                    1
                    + (document_count + 1)
                    / (document_frequency + 1)
                )
                + 1
            )

        self.vectors = [
            self._vector(tokenize(chunk["text"]))
            for chunk in self.chunks
        ]

    def _vector(self, tokens):
        """Create a normalized TF-IDF-style vector."""

        counts = Counter(tokens)

        vector = {
            term: (
                1 + math.log(count)
            ) * self.idf.get(term, 0.0)

            for term, count in counts.items()
            if term in self.idf
        }

        norm = math.sqrt(
            sum(value * value for value in vector.values())
        )

        if norm == 0:
            norm = 1.0

        return {
            term: value / norm
            for term, value in vector.items()
        }

    def _similarity(self, query, vector):
        """Calculate cosine similarity between query and document."""

        query_vector = self._vector(tokenize(query))

        return sum(
            value * vector.get(term, 0.0)
            for term, value in query_vector.items()
        )

    def retrieve(self, question, top_k=3):
        """
        Retrieve the most relevant knowledge-base sections.
        Results are sorted by similarity score.
        """

        scored_chunks = [
            (
                self._similarity(question, vector),
                chunk
            )
            for vector, chunk in zip(
                self.vectors,
                self.chunks
            )
        ]

        scored_chunks.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return scored_chunks[:top_k]

    def _best_sentence(self, question, passage):
        """
        Extract the single most relevant sentence from a passage.

        Returns None when no sentence has enough lexical overlap
        with the question.
        """

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                passage.strip()
            )
            if sentence.strip()
        ]

        if not sentences:
            return None

        ranked_sentences = sorted(
            (
                (
                    self._similarity(
                        question,
                        self._vector(tokenize(sentence))
                    ),
                    sentence
                )
                for sentence in sentences
            ),
            key=lambda item: item[0],
            reverse=True
        )

        best_score, best_sentence = ranked_sentences[0]

        # Reject weak sentence matches instead of returning
        # an unrelated sentence from a broadly relevant paragraph.
        if best_score < self.threshold * 0.55:
            return None

        return best_sentence

    def answer(self, question):
        """
        Answer a question using only retrieved knowledge-base text.

        Returns:
            {
                "answered": bool,
                "answer": str,
                "sources": list
            }
        """

        question = question.strip()

        if not question:
            return {
                "answered": False,
                "answer": "Please enter a question.",
                "sources": []
            }

        # Step 1: Retrieve relevant passages.
        retrieved = self.retrieve(question)

        # Step 2: Abstain if no passage is sufficiently relevant.
        if (
            not retrieved
            or retrieved[0][0] < self.threshold
        ):
            return {
                "answered": False,
                "answer": ABSTENTION_MESSAGE,
                "sources": []
            }

        best_score, best_chunk = retrieved[0]

        # Step 3: Extract one relevant sentence.
        sentence = self._best_sentence(
            question,
            best_chunk["text"]
        )

        # Step 4: Abstain if the sentence match is too weak.
        if not sentence:
            return {
                "answered": False,
                "answer": ABSTENTION_MESSAGE,
                "sources": []
            }

        # Step 5: Include source evidence for transparency.
        sources = []

        for score, chunk in retrieved[:2]:

            if score >= max(
                self.threshold,
                best_score * 0.55
            ):
                sources.append({
                    "file": chunk["file"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "score": score
                })

        # Step 6: Return the focused answer, not the full paragraph.
        return {
            "answered": True,
            "answer": sentence,
            "sources": sources
        }