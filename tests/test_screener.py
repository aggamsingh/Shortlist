import unittest
from unittest.mock import MagicMock, patch
import os

# Setup environment variables for testing
os.environ["API_KEY"] = "test-secret-key"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["QDRANT_COLLECTION"] = "test-resumes"
os.environ["DEFAULT_TOP_K"] = "5"
os.environ["RETRIEVAL_TOP_N"] = "10"

# Start global mocks before any project code import to prevent network/DB requests
patcher_transformer = patch('sentence_transformers.SentenceTransformer')
mock_transformer_cls = patcher_transformer.start()
mock_transformer_instance = MagicMock()
mock_transformer_instance.get_sentence_embedding_dimension.return_value = 384
mock_transformer_instance.encode.return_value = [[0.1] * 384]
mock_transformer_cls.return_value = mock_transformer_instance

patcher_qdrant = patch('qdrant_client.QdrantClient')
mock_qdrant_cls = patcher_qdrant.start()
mock_qdrant_instance = MagicMock()
mock_qdrant_cls.return_value = mock_qdrant_instance

from indexer.parser import clean_text, extract_years_of_experience, chunk_cv
from indexer.run import clean_candidate_name, extract_location
from api.main import app
import api.main as main_module

from fastapi.testclient import TestClient

class TestParserAndUtils(unittest.TestCase):
    def test_clean_text(self):
        text = "Hello    World!   \n New Line  "
        self.assertEqual(clean_text(text), "Hello World! New Line")

    def test_extract_years_of_experience(self):
        cv1 = "I have 5 years of experience as a python developer."
        cv2 = "Total Experience: 12 yrs"
        cv3 = "No experience mentioned here."
        
        self.assertEqual(extract_years_of_experience(cv1), 5)
        self.assertEqual(extract_years_of_experience(cv2), 12)
        self.assertEqual(extract_years_of_experience(cv3), 0)

    def test_extract_location(self):
        text_delhi = "I live in Delhi, India."
        text_bengaluru = "Worked at a startup in Bengaluru."
        text_unknown = "Remote worker."
        
        self.assertEqual(extract_location(text_delhi), "Delhi")
        self.assertEqual(extract_location(text_bengaluru), "Bangalore")
        self.assertEqual(extract_location(text_unknown), "Unknown")

    def test_clean_candidate_name(self):
        self.assertEqual(clean_candidate_name("Jane_Doe_Resume_2026.pdf"), "Jane Doe")
        self.assertEqual(clean_candidate_name("john-smith-cv.docx"), "John Smith")
        self.assertEqual(clean_candidate_name("cv_developer.pdf"), "Developer")

    def test_chunk_cv_section_split(self):
        text = """
Jane Doe
Jane@example.com

Summary
Passionate engineer with experience.

Experience
Worked at Google for 5 years as backend developer.
Worked at Meta for 2 years.

Skills
Python, FastAPI, Qdrant, Docker
"""
        chunks = chunk_cv(text)
        self.assertTrue(len(chunks) >= 3)
        self.assertTrue(any("SUMMARY" in c for c in chunks))
        self.assertTrue(any("EXPERIENCE" in c for c in chunks))
        self.assertTrue(any("SKILLS" in c for c in chunks))

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        # Setup mocks for API module singletons
        main_module.embedder = MagicMock()
        main_module.embedder.embed_text.return_value = [0.1] * 384
        main_module.retriever = MagicMock()
        main_module.reranker = MagicMock()
        self.client = TestClient(app)

    def test_health_check_healthy(self):
        main_module.retriever.client = MagicMock()
        main_module.retriever.client.get_collections.return_value = None
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_health_check_unhealthy(self):
        main_module.retriever.client = MagicMock()
        main_module.retriever.client.get_collections.side_effect = Exception("Connection Refused")
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["status"], "unhealthy")

    def test_api_key_auth_required(self):
        payload = {
            "job_description": "We need a Python developer.",
            "top_k": 5
        }
        # Request without header
        response = self.client.post("/api/v1/screen", json=payload)
        self.assertEqual(response.status_code, 401)

        # Request with invalid header
        response = self.client.post("/api/v1/screen", json=payload, headers={"X-API-Key": "wrong-key"})
        self.assertEqual(response.status_code, 401)

        # Request with valid header
        main_module.retriever.search_candidates.return_value = [
            {
                "candidate_id": "cand-1",
                "name": "Jane Doe",
                "cv_path": "/app/cvs/jane_doe.pdf",
                "years_of_experience": 5,
                "location": "Delhi",
                "score": 0.85,
                "resume_summary": "Resume summary text"
            }
        ]
        main_module.reranker.rerank.return_value = [
            {
                "candidate_id": "cand-1",
                "name": "Jane Doe",
                "score": 0.95,
                "match_reasoning": "Strong experience matching description.",
                "cv_path": "/app/cvs/jane_doe.pdf"
            }
        ]

        response = self.client.post("/api/v1/screen", json=payload, headers={"X-API-Key": "test-secret-key"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(len(data["candidates"]), 1)
        self.assertEqual(data["candidates"][0]["name"], "Jane Doe")
        self.assertEqual(data["candidates"][0]["score"], 0.95)

# Stop patchers when file exits, but keep active during tests
def tearDownModule():
    patcher_transformer.stop()
    patcher_qdrant.stop()

if __name__ == "__main__":
    unittest.main()
