"""BM25 sparse encoding and hybrid retrieval tests.

Hybrid retrieval exists because the evaluation showed dense search was the
bottleneck: the reranker already promoted every relevant candidate it was given,
so the remaining loss was candidates retrieval never surfaced. These tests pin
the properties that make BM25 worth adding -- exact rare terms outrank common
ones -- and the invariant that a missing BM25 state degrades to dense-only
search rather than failing a request.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from api.retriever import SPARSE_VECTOR_NAME, CVRetriever
from indexer.sparse import BM25Encoder, token_id, tokenize

CORPUS = [
    "Python backend engineer with FastAPI and Qdrant vector database experience",
    "Python backend engineer with Django and PostgreSQL experience",
    "Frontend developer with React and TypeScript experience",
    "DevOps engineer with Kubernetes and Terraform on AWS",
]


class TokenizerTest(unittest.TestCase):
    def test_keeps_technical_tokens_intact(self):
        """Splitting c++ or node.js into fragments would destroy the signal."""
        self.assertIn("c++", tokenize("Experienced in C++ development"))
        self.assertIn("node.js", tokenize("Built with node.js"))
        self.assertIn("c#", tokenize("C# and .NET"))

    def test_lowercases_and_drops_stopwords(self):
        self.assertEqual(tokenize("The Python AND the Django"), ["python", "django"])

    def test_drops_single_characters(self):
        self.assertEqual(tokenize("a b python"), ["python"])

    def test_empty_input(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])

    def test_token_id_is_stable_across_calls(self):
        """hash() is salted per process and would break the index between runs."""
        self.assertEqual(token_id("qdrant"), token_id("qdrant"))
        self.assertNotEqual(token_id("qdrant"), token_id("pinecone"))


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.enc = BM25Encoder().fit(CORPUS)

    def score(self, query: str, doc: str) -> float:
        qi, qv = self.enc.encode_query(query)
        di, dv = self.enc.encode_document(doc)
        dmap = dict(zip(di, dv))
        return sum(w * dmap.get(i, 0.0) for i, w in zip(qi, qv))

    def test_fit_populates_statistics(self):
        self.assertTrue(self.enc.is_fitted)
        self.assertEqual(self.enc.doc_count, len(CORPUS))
        self.assertGreater(self.enc.avgdl, 0)

    def test_rare_terms_outweigh_common_ones(self):
        """The entire reason BM25 helps here: exact, low-frequency tech terms."""
        self.assertGreater(self.enc.idf["qdrant"], self.enc.idf["python"])

    def test_exact_term_match_ranks_first(self):
        scores = {doc: self.score("Qdrant vector database", doc) for doc in CORPUS}
        best = max(scores, key=scores.get)
        self.assertIn("Qdrant", best)

    def test_documents_without_query_terms_score_zero(self):
        self.assertEqual(self.score("Kubernetes Terraform", CORPUS[2]), 0.0)

    def test_unseen_query_terms_are_dropped(self):
        indices, _ = self.enc.encode_query("kubernetes zzzznotacorpusterm")
        self.assertEqual(len(indices), 1)

    def test_empty_and_stopword_only_inputs(self):
        self.assertEqual(self.enc.encode_document(""), ([], []))
        self.assertEqual(self.enc.encode_query(""), ([], []))
        self.assertEqual(self.enc.encode_query("the and of"), ([], []))

    def test_unfitted_encoder_is_inert(self):
        """An unfitted encoder must be usable, so callers can fall back cleanly."""
        blank = BM25Encoder()
        self.assertFalse(blank.is_fitted)
        self.assertEqual(blank.encode_document("python"), ([], []))
        self.assertEqual(blank.encode_query("python"), ([], []))

    def test_fit_on_empty_corpus_does_not_raise(self):
        self.assertFalse(BM25Encoder().fit([]).is_fitted)

    def test_indices_are_sorted(self):
        """Qdrant expects sparse indices in ascending order."""
        indices, _ = self.enc.encode_document(CORPUS[0])
        self.assertEqual(indices, sorted(indices))

    def test_round_trip_through_disk(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "bm25.json"
            self.enc.save(str(path))
            loaded = BM25Encoder.load(str(path))
            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.doc_count, self.enc.doc_count)
            self.assertAlmostEqual(loaded.avgdl, self.enc.avgdl)
            self.assertEqual(loaded.encode_query("qdrant"), self.enc.encode_query("qdrant"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_missing_file_returns_unfitted(self):
        """A missing state file is a degraded mode, not an error."""
        self.assertFalse(BM25Encoder.load("/nonexistent/bm25.json").is_fitted)

    def test_load_corrupt_file_returns_unfitted(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "bad.json"
            path.write_text("not json at all", encoding="utf-8")
            self.assertFalse(BM25Encoder.load(str(path)).is_fitted)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class HybridRetrievalTest(unittest.TestCase):
    """Hybrid search over a real embedded Qdrant with dense + sparse vectors."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hybrid_test_"))
        cls.client = QdrantClient(path=str(cls.tmp / "q"))
        cls.encoder = BM25Encoder().fit(CORPUS)
        cls.client.create_collection(
            "hybrid",
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )
        # Dense vectors are deliberately misleading: the Qdrant CV (id 1) is given
        # the WEAKEST dense score for the query below, so only the sparse branch
        # can surface it. If hybrid works, it still ranks first.
        dense = {1: [0.1, 0.9, 0.0, 0.0], 2: [0.9, 0.1, 0.0, 0.0],
                 3: [0.8, 0.2, 0.0, 0.0], 4: [0.7, 0.3, 0.0, 0.0]}
        points = []
        for i, text in enumerate(CORPUS, start=1):
            idx, val = cls.encoder.encode_document(text)
            points.append(models.PointStruct(
                id=i,
                vector={"": dense[i], SPARSE_VECTOR_NAME: models.SparseVector(indices=idx, values=val)},
                payload={"candidate_id": f"c{i}", "name": f"Candidate {i}",
                         "cv_path": f"/{i}.pdf", "chunk_text": text,
                         "years_of_experience": 5, "location": "Delhi"},
            ))
        cls.client.upsert("hybrid", points=points)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _retriever(self, fitted=True):
        r = CVRetriever(client=self.client)
        r.collection_name = "hybrid"
        r.hybrid_enabled = True
        r.bm25 = self.encoder if fitted else BM25Encoder()
        return r

    def test_sparse_branch_rescues_a_dense_miss(self):
        """The payoff: an exact term match that dense search ranks last."""
        query_vector = [0.9, 0.1, 0.0, 0.0]  # points away from the Qdrant CV

        dense_only = self._retriever(fitted=False).search_candidates(
            query_vector=query_vector, filters=None, top_n=4, query_text="Qdrant"
        )
        hybrid = self._retriever().search_candidates(
            query_vector=query_vector, filters=None, top_n=4, query_text="Qdrant"
        )

        self.assertNotEqual(dense_only[0]["candidate_id"], "c1",
                            "dense alone should not surface the Qdrant CV first")
        self.assertEqual(hybrid[0]["candidate_id"], "c1",
                         "hybrid should promote the exact term match")

    def test_unfitted_bm25_falls_back_to_dense(self):
        """A missing BM25 state must degrade, never fail."""
        results = self._retriever(fitted=False).search_candidates(
            query_vector=[0.9, 0.1, 0.0, 0.0], filters=None, top_n=4, query_text="Qdrant"
        )
        self.assertEqual(len(results), 4)

    def test_missing_query_text_falls_back_to_dense(self):
        results = self._retriever().search_candidates(
            query_vector=[0.9, 0.1, 0.0, 0.0], filters=None, top_n=4, query_text=None
        )
        self.assertEqual(len(results), 4)

    def test_query_with_no_corpus_terms_falls_back_to_dense(self):
        results = self._retriever().search_candidates(
            query_vector=[0.9, 0.1, 0.0, 0.0], filters=None, top_n=4,
            query_text="zzzznotacorpusterm",
        )
        self.assertEqual(len(results), 4)

    def test_hybrid_returns_distinct_candidates(self):
        results = self._retriever().search_candidates(
            query_vector=[0.5, 0.5, 0.0, 0.0], filters=None, top_n=4,
            query_text="Python engineer",
        )
        ids = [c["candidate_id"] for c in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_scores_stay_within_the_response_contract(self):
        """RRF scores are a different quantity from cosine and must be clamped."""
        results = self._retriever().search_candidates(
            query_vector=[0.5, 0.5, 0.0, 0.0], filters=None, top_n=4,
            query_text="Python Qdrant Kubernetes",
        )
        for c in results:
            self.assertGreaterEqual(c["score"], 0.0)
            self.assertLessEqual(c["score"], 1.0)

    def test_filters_still_apply_under_hybrid(self):
        from api.models import ScreeningFilters

        results = self._retriever().search_candidates(
            query_vector=[0.5, 0.5, 0.0, 0.0],
            filters=ScreeningFilters(min_experience=60, location=None),
            top_n=4, query_text="Python",
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
