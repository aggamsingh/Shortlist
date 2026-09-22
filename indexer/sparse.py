"""BM25 sparse encoding for hybrid retrieval.

Why this exists
---------------
The evaluation showed dense retrieval was the bottleneck: on the within-role
query set the reranker already promoted every relevant candidate it was given
(recall@5 reached pool recall exactly), so the remaining loss is candidates that
retrieval never surfaced. Pool recall sat at 0.822, meaning ~18% of relevant
candidates never reached the reranker at all.

Embeddings are weak at exact, low-frequency technical tokens -- "Qdrant",
"Pinecone", "asyncio", "Terraform" -- which is precisely what the hard queries
turn on. BM25 is strong there and weak at paraphrase, so the two are
complementary. Scores are fused by Qdrant's native Reciprocal Rank Fusion.

Design
------
Document vectors carry the term-frequency saturation weight; query vectors carry
the IDF. Their dot product is then the BM25 score, which is what Qdrant's sparse
index computes. This split means IDF statistics only have to be applied at query
time, but they must be derived from the whole corpus at index time -- hence the
fitted state saved alongside the index.

Token ids come from CRC32 rather than a stored vocabulary, so no vocab file has
to stay in sync. Python's built-in hash() is salted per process and would give
different ids on every run, which would silently break retrieval.
"""

import json
import math
import re
import zlib
from collections import Counter
from pathlib import Path

from indexer.utils import get_logger

logger = get_logger("indexer.sparse")

# Standard BM25 parameters. k1 controls term-frequency saturation, b controls
# length normalisation. These are the usual defaults and were not tuned.
K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# Deliberately small. Aggressive stopword removal hurts here: words like "lead"
# or "own" carry signal in a resume, and IDF already discounts common terms.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    with by from as is are was were be been being it its his her their our your
    i you he she we they them us me my
    """.split()
)


def tokenize(text: str) -> list:
    """Lowercase and split into terms, keeping technical tokens intact.

    The character class keeps '+', '#' and '.' so that c++, c# and node.js do
    not get shredded into meaningless fragments.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t.strip(".") for t in tokens
            if len(t.strip(".")) > 1 and t.strip(".") not in STOPWORDS]


def token_id(token: str) -> int:
    """Stable, process-independent id for a token.

    CRC32 is not cryptographic, but collisions here only blur two unrelated
    terms in the sparse index, and it is deterministic across runs and machines,
    which hash() is not.
    """
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


class BM25Encoder:
    """Fits IDF/average-length statistics, then encodes documents and queries."""

    def __init__(self):
        self.idf = {}
        self.avgdl = 0.0
        self.doc_count = 0

    @property
    def is_fitted(self) -> bool:
        return bool(self.idf) and self.avgdl > 0

    def fit(self, documents) -> "BM25Encoder":
        """Compute IDF and average document length over the whole corpus."""
        doc_freq = Counter()
        total_len = 0
        count = 0

        for text in documents:
            tokens = tokenize(text)
            count += 1
            total_len += len(tokens)
            doc_freq.update(set(tokens))

        if count == 0:
            logger.warning("BM25Encoder.fit received no documents; encoder stays unfitted.")
            return self

        self.doc_count = count
        self.avgdl = total_len / count
        # Probabilistic IDF with the +1 guard, so a term appearing in every
        # document gets a small positive weight rather than a negative one.
        self.idf = {
            token: math.log(1 + (count - df + 0.5) / (df + 0.5))
            for token, df in doc_freq.items()
        }
        logger.info(
            f"BM25 fitted on {count} documents | vocabulary {len(self.idf)} terms | "
            f"avg length {self.avgdl:.1f} tokens"
        )
        return self

    def encode_document(self, text: str) -> tuple:
        """Return (indices, values) carrying term-frequency saturation weights."""
        tokens = tokenize(text)
        if not tokens or not self.is_fitted:
            return [], []

        length = len(tokens)
        norm = K1 * (1 - B + B * length / self.avgdl)

        weights = {}
        for token, tf in Counter(tokens).items():
            # Only terms seen during fitting can ever be matched by a query.
            if token not in self.idf:
                continue
            weights[token_id(token)] = tf * (K1 + 1) / (tf + norm)

        if not weights:
            return [], []
        indices = sorted(weights)
        return indices, [weights[i] for i in indices]

    def encode_query(self, text: str) -> tuple:
        """Return (indices, values) carrying IDF weights.

        Dotted with a document vector from encode_document, this yields the BM25
        score for the pair.
        """
        tokens = tokenize(text)
        if not tokens or not self.is_fitted:
            return [], []

        weights = {}
        for token in set(tokens):
            idf = self.idf.get(token)
            # An unseen term matches no document, so omitting it changes nothing
            # except the size of the query vector.
            if idf is None:
                continue
            weights[token_id(token)] = idf

        if not weights:
            return [], []
        indices = sorted(weights)
        return indices, [weights[i] for i in indices]

    # ---- persistence ----

    def save(self, path: str) -> None:
        """Persist fitted statistics next to the index so the API can reuse them."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"doc_count": self.doc_count, "avgdl": self.avgdl, "idf": self.idf},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(f"Saved BM25 state ({len(self.idf)} terms) to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Encoder":
        """Load fitted statistics. Returns an UNFITTED encoder if unavailable.

        An unfitted encoder is a valid state, not an error: the retriever then
        falls back to dense-only search rather than failing the request.
        """
        encoder = cls()
        file = Path(path)
        if not file.exists():
            logger.info(f"No BM25 state at {path}; hybrid retrieval unavailable.")
            return encoder
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            encoder.idf = data.get("idf", {})
            encoder.avgdl = data.get("avgdl", 0.0)
            encoder.doc_count = data.get("doc_count", 0)
            logger.info(f"Loaded BM25 state ({len(encoder.idf)} terms) from {path}")
        except Exception as e:
            logger.warning(f"Could not read BM25 state from {path}: {e}. Dense-only search.")
        return encoder
