import os
import json
import google.generativeai as genai
from groq import Groq
from indexer.utils import get_logger

logger = get_logger("api.reranker")

class CVReranker:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        if not self.gemini_key and not self.groq_key:
            logger.error("No LLM provider API key found in environment (GEMINI_API_KEY or GROQ_API_KEY).")
        
        # Configure Gemini if key exists
        if self.gemini_key:
            logger.info("Initializing Gemini API client for reranking.")
            genai.configure(api_key=self.gemini_key)
        
        # Configure Groq if key exists
        if self.groq_key:
            logger.info("Initializing Groq API client for reranking.")
            self.groq_client = Groq(api_key=self.groq_key)

    def _rerank_with_gemini(self, jd: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using Gemini 1.5 Flash."""
        prompt = self._build_prompt(jd, candidates)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        logger.info("Sending reranking request to Gemini API...")
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        try:
            result = json.loads(response.text)
            return result.get("rankings", [])
        except Exception as e:
            logger.error(f"Failed to parse JSON response from Gemini: {e}. Raw response: {response.text}")
            raise RuntimeError("Gemini did not return valid JSON rankings.")

    def _rerank_with_groq(self, jd: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using Groq with Llama 3 70B."""
        prompt = self._build_prompt(jd, candidates)
        
        logger.info("Sending reranking request to Groq API...")
        completion = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        response_text = completion.choices[0].message.content
        try:
            result = json.loads(response_text)
            return result.get("rankings", [])
        except Exception as e:
            logger.error(f"Failed to parse JSON response from Groq: {e}. Raw response: {response_text}")
            raise RuntimeError("Groq did not return valid JSON rankings.")

    def _build_prompt(self, jd: str, candidates: list[dict]) -> str:
        candidates_str = ""
        for i, c in enumerate(candidates):
            candidates_str += f"""
Candidate Index: {i}
Candidate ID: {c['candidate_id']}
Candidate Name: {c['name']}
Resume Text/Summary:
{c['resume_summary']}
----------------------
"""
        
        prompt = f"""You are an expert recruiter assistant. Evaluate and score the following candidates against the provided Job Description.

Job Description:
{jd}

Candidates:
{candidates_str}

Evaluate each candidate. Provide:
1. "candidate_id": the unique ID of the candidate.
2. "score": a floating point value between 0.00 and 1.00 indicating the fit.
3. "match_reasoning": a concise, single-sentence summary of why they match or mismatch (max 120 characters).

Return your response in JSON format matching this schema:
{{
  "rankings": [
    {{
      "candidate_id": "candidate_id",
      "score": 0.95,
      "match_reasoning": "Strong python developer with 4 years of Django and vector search experience matching the job description."
    }}
  ]
}}
Ensure every candidate in the input list is evaluated and included in the output.
"""
        return prompt

    def rerank(self, jd: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Rerank candidates based on Job Description and return top_k candidates."""
        if not candidates:
            return []
            
        if not self.gemini_key and not self.groq_key:
            raise ValueError("No LLM API keys provided. Please set GEMINI_API_KEY or GROQ_API_KEY in the environment.")
            
        # Rerank using the available client
        rankings = []
        if self.gemini_key:
            try:
                rankings = self._rerank_with_gemini(jd, candidates)
            except Exception as e:
                logger.warning(f"Gemini reranking failed: {e}. Trying Groq fallback if configured...")
                if self.groq_key:
                    rankings = self._rerank_with_groq(jd, candidates)
                else:
                    raise
        elif self.groq_key:
            rankings = self._rerank_with_groq(jd, candidates)

        # Merge LLM score and reasoning back with candidate metadata (name, cv_path, etc.)
        rankings_map = {r["candidate_id"]: r for r in rankings if "candidate_id" in r}
        
        reranked_results = []
        for cand in candidates:
            cand_id = cand["candidate_id"]
            if cand_id in rankings_map:
                llm_data = rankings_map[cand_id]
                reranked_results.append({
                    "candidate_id": cand_id,
                    "name": cand["name"],
                    "score": float(llm_data.get("score", cand["score"])),
                    "match_reasoning": llm_data.get("match_reasoning", "Semantic match in resume indexing."),
                    "cv_path": cand["cv_path"]
                })
            else:
                logger.warning(f"Candidate {cand['name']} ({cand_id}) was skipped by LLM reranker. Applying fallback.")
                reranked_results.append({
                    "candidate_id": cand_id,
                    "name": cand["name"],
                    "score": float(cand["score"]),
                    "match_reasoning": "Semantic search retrieval match.",
                    "cv_path": cand["cv_path"]
                })

        # Sort by reranked score descending
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results[:top_k]
