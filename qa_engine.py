"""
Context-Aware Q&A Assistant

Features:
- Retrieves relevant information from local knowledge-base files.
- Uses TF-IDF-style keyword matching.
- Expands common question wording.
- Extracts a relevant sentence instead of returning whole paragraphs.
- Searches all knowledge-base sections for casual-leave timing questions.
- Clearly labels annual-leave fallback information.
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
        Load Markdown and text files from a knowledge-base directory.
        Each Markdown heading section becomes a searchable chunk.
        """
        root = Path(directory)

        if not root.exists():
            raise FileNotFoundError(
                f"Knowledge-base directory not found: {root.resolve()}"
            )

        if not root.is_dir():
            raise NotADirectoryError(
                f"Expected a directory: {root.resolve()}"
            )

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

            # Split at Markdown headings.
            blocks = re.split(
                r"(?m)(?=^#{1,2}\s+)",
                content
            )

            for block in blocks:
                lines = block.strip().splitlines()

                if not lines:
                    continue

                if re.match(r"^#{1,2}\s+", lines[0]):
                    section = re.sub(
                        r"^#{1,2}\s+",
                        "",
                        lines[0]
                    ).strip() or section

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
                    1 + (document_count + 1)
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

    def _query_tokens(self, question):
        """Expand common question wording for lexical matching."""

        tokens = tokenize(question)

        stop_words = {
            "how", "far", "ahead", "can", "i", "a", "an", "the",
            "is", "are", "do", "does", "for", "to", "of", "in",
            "what", "when", "where", "should", "could", "would",
            "me", "my", "be", "it", "much", "please", "tell"
        }

        tokens = [
            token
            for token in tokens
            if token not in stop_words
        ]

        synonyms = {
            "apply": ["request", "submit"],
            "applied": ["requested", "submitted"],
            "applying": ["request", "submit"],
            "application": ["request", "submit"],
            "applications": ["requests"],
            "submitted": ["submit", "request"],
            "submitting": ["submit", "request"],
            "requested": ["request", "submit"],
            "requests": ["request", "submit"],
            "early": ["advance"],
            "before": ["advance"],
            "earlier": ["advance"],
            "deadline": ["period"],
        }

        expanded = list(tokens)

        for token in tokens:
            expanded.extend(synonyms.get(token, []))

        return list(dict.fromkeys(expanded))

    def _similarity(self, query, vector):
        """Calculate cosine similarity using expanded query terms."""

        query_vector = self._vector(
            self._query_tokens(query)
        )

        return sum(
            value * vector.get(term, 0.0)
            for term, value in query_vector.items()
        )

    def retrieve(self, question, top_k=5):
        """Return the highest-scoring knowledge-base sections."""

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

    def _is_timing_question(self, question):
        """Check whether the user asks about timing or notice periods."""

        question_lower = question.lower()

        timing_phrases = [
            "how far ahead",
            "how far in advance",
            "how many days before",
            "how early",
            "when should i apply",
            "when can i apply",
            "when do i need to apply",
            "how much notice",
            "notice period",
            "advance notice",
            "how long before",
        ]

        return any(
            phrase in question_lower
            for phrase in timing_phrases
        )

    def _best_sentence(self, question, passage):
        """Extract the most relevant sentence from a passage."""

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

        if self._is_timing_question(question):
            action_words = [
                "request",
                "requests",
                "requested",
                "submit",
                "submitted",
                "submitting",
                "apply",
                "application",
                "notify",
            ]

            timing_words = [
                "advance",
                "working days",
                "days before",
                "prior to",
                "at least",
                "no later than",
                "as early as practical",
                "before the",
            ]

            relevant_sentences = [
                sentence
                for sentence in sentences
                if any(
                    word in sentence.lower()
                    for word in action_words
                )
                and any(
                    word in sentence.lower()
                    for word in timing_words
                )
            ]

            if not relevant_sentences:
                return None

            sentences = relevant_sentences

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

        if best_score < self.threshold * 0.55:
            return None

        return best_sentence

    def _requested_leave_type(self, question):
        """Identify an explicitly requested leave category."""

        question_lower = question.lower()

        if re.search(r"\bcasual leave\b", question_lower):
            return "casual"

        if re.search(r"\bannual leave\b", question_lower):
            return "annual"

        if re.search(r"\bsick leave\b", question_lower):
            return "sick"

        if re.search(r"\bparental leave\b", question_lower):
            return "parental"

        return None

    def _chunk_leave_type(self, chunk):
        """Identify the leave category documented in a section."""

        section = chunk.get("section", "").lower()
        text = chunk.get("text", "").lower()

        if "casual leave" in section:
            return "casual"

        if "annual leave" in section:
            return "annual"

        if "sick leave" in section:
            return "sick"

        if "parental leave" in section:
            return "parental"

        # Avoid treating a section as annual leave merely because
        # it contains a generic mention of the word "leave".
        if re.search(r"\bannual leave\b", text):
            return "annual"

        if re.search(r"\bsick leave\b", text):
            return "sick"

        if re.search(r"\bparental leave\b", text):
            return "parental"

        if re.search(r"\bcasual leave\b", text):
            return "casual"

        return None

    def _find_annual_leave_timing_rule(self):
        """
        Search every chunk for the documented annual-leave notice rule.
        This deliberately does not depend on retrieval ranking.
        """

        annual_chunks = [
            chunk
            for chunk in self.chunks
            if chunk.get("section", "").strip().lower()
            == "annual leave"
        ]

        for chunk in annual_chunks:
            sentences = [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    chunk["text"].strip()
                )
                if sentence.strip()
            ]

            for sentence in sentences:
                sentence_lower = sentence.lower()

                has_request_action = any(
                    word in sentence_lower
                    for word in [
                        "request",
                        "requests",
                        "submit",
                        "submitted",
                        "apply",
                        "application",
                    ]
                )

                has_timing_rule = any(
                    phrase in sentence_lower
                    for phrase in [
                        "in advance",
                        "working days in advance",
                        "days before",
                        "prior to",
                        "advance notice",
                    ]
                )

                if has_request_action and has_timing_rule:
                    return chunk, sentence

        return None, None

    def answer(self, question):
        """
        Answer a question using only supplied knowledge-base text.

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

        if not self.chunks:
            return {
                "answered": False,
                "answer": ABSTENTION_MESSAGE,
                "sources": []
            }

        requested_leave = self._requested_leave_type(question)

        # Handle casual-leave timing questions before normal retrieval.
        if (
            requested_leave == "casual"
            and self._is_timing_question(question)
        ):
            chunk, sentence = self._find_annual_leave_timing_rule()

            if chunk and sentence:
                score = self._similarity(
                    question,
                    self._vector(tokenize(chunk["text"]))
                )

                return {
                    "answered": True,
                    "answer": (
                        f"{sentence} "
                        "Note: This rule is documented under Annual "
                        "Leave. The knowledge base does not confirm "
                        "whether it also applies to Casual Leave."
                    ),
                    "sources": [{
                        "file": chunk["file"],
                        "section": chunk["section"],
                        "text": chunk["text"],
                        "score": score
                    }]
                }

            return {
                "answered": False,
                "answer": (
                    "The knowledge base does not specify how far in "
                    "advance casual leave must be requested."
                ),
                "sources": []
            }

        # Normal retrieval for other questions.
        retrieved = self.retrieve(question, top_k=5)

        if not retrieved or retrieved[0][0] < self.threshold:
            return {
                "answered": False,
                "answer": ABSTENTION_MESSAGE,
                "sources": []
            }

        # For other explicit leave types, use matching sections only.
        if requested_leave:
            matching_chunks = [
                (score, chunk)
                for score, chunk in retrieved
                if self._chunk_leave_type(chunk) == requested_leave
            ]

            if not matching_chunks:
                return {
                    "answered": False,
                    "answer": (
                        "The provided knowledge base does not specify "
                        f"a separate policy for {requested_leave} leave."
                    ),
                    "sources": []
                }

            retrieved = matching_chunks

        # Find a relevant sentence across retrieved sections.
        candidates = []

        for score, chunk in retrieved:
            sentence = self._best_sentence(
                question,
                chunk["text"]
            )

            if sentence:
                sentence_score = self._similarity(
                    question,
                    self._vector(tokenize(sentence))
                )

                candidates.append({
                    "chunk_score": score,
                    "sentence_score": sentence_score,
                    "sentence": sentence,
                    "chunk": chunk
                })

        if not candidates:
            return {
                "answered": False,
                "answer": ABSTENTION_MESSAGE,
                "sources": []
            }

        candidates.sort(
            key=lambda item: (
                item["sentence_score"],
                item["chunk_score"]
            ),
            reverse=True
        )

        best = candidates[0]
        chunk = best["chunk"]

        return {
            "answered": True,
            "answer": best["sentence"],
            "sources": [{
                "file": chunk["file"],
                "section": chunk["section"],
                "text": chunk["text"],
                "score": best["chunk_score"]
            }]
        }
