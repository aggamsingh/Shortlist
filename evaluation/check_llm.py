"""Verify an LLM provider actually works, before spending a full eval run on it.

Run this first after putting a key in .env:

    python -m evaluation.check_llm

It makes ONE real API call per configured provider with three tiny candidates
whose correct ranking is obvious, then reports exactly what came back. Model ids
get retired and keys get scoped wrong; this tells you which of those happened
instead of surfacing it as an empty rerank ten minutes into an evaluation.

Exit code is 0 if at least one provider answered correctly, 1 otherwise.
"""

import sys

from dotenv import load_dotenv

from api.reranker import CVReranker

# Deliberately unambiguous: a competent reranker must put c_vector first and
# c_frontend last. If a provider cannot do this, the problem is the provider or
# the model id, not the pipeline.
JOB_DESCRIPTION = (
    "Senior Python Backend Engineer. You must have hands-on production "
    "experience with a vector database and embedding-based semantic retrieval. "
    "This is the core requirement of the role."
)

CANDIDATES = [
    {
        "candidate_id": "c_frontend",
        "name": "Frontend Developer",
        "cv_path": "/demo/frontend.pdf",
        "score": 0.61,  # deliberately the HIGHEST vector score
        "resume_summary": (
            "Frontend developer with 3 years building React and TypeScript "
            "interfaces. Maintains some Python build scripts."
        ),
    },
    {
        "candidate_id": "c_vector",
        "name": "Retrieval Engineer",
        "cv_path": "/demo/vector.pdf",
        "score": 0.55,
        "resume_summary": (
            "Backend engineer who built a production RAG pipeline over 5M "
            "documents using a Pinecone vector database, owning chunking and "
            "embedding strategy, served through FastAPI."
        ),
    },
    {
        "candidate_id": "c_django",
        "name": "Django Developer",
        "cv_path": "/demo/django.pdf",
        "score": 0.58,
        "resume_summary": (
            "Python backend engineer with 5 years of Django REST Framework and "
            "PostgreSQL experience. No vector search work."
        ),
    },
]


def check_provider(reranker: CVReranker, provider: str) -> bool:
    """Make one real call through a single provider and report the outcome."""
    label = {"gemini": reranker.gemini_model, "groq": reranker.groq_model}[provider]
    print(f"\n--- {provider} ({label}) ---")

    call = {
        "gemini": reranker._rerank_with_gemini,
        "groq": reranker._rerank_with_groq,
    }[provider]

    try:
        rankings = call(JOB_DESCRIPTION, CANDIDATES)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  Check that the API key is valid and the model id still exists.")
        return False

    if not rankings:
        print("  FAILED: provider returned no rankings.")
        return False

    print(f"  raw rankings returned: {len(rankings)}")
    merged = reranker._merge(CANDIDATES, rankings, top_k=len(CANDIDATES))
    for row in merged:
        print(f"    {row['name']:22} {row['score']:.2f}  {row['match_reasoning'][:60]}")

    order = [row["candidate_id"] for row in merged]
    correct = order[0] == "c_vector" and order[-1] == "c_frontend"

    if correct:
        print("  OK: ranked the vector-database candidate first and the frontend last,")
        print("      overriding the vector scores (frontend had the highest cosine).")
    else:
        print(f"  WARNING: provider answered but ranked them {order}.")
        print("      Expected c_vector first and c_frontend last.")
        print("      The pipeline works; the model's judgement is the weak part here.")
    return correct


def main() -> int:
    # Loaded here rather than at import: calling load_dotenv() as an import side
    # effect injects real keys into os.environ for any process that merely
    # imports this module, which silently broke tests asserting "no key set".
    load_dotenv(dotenv_path=".env")

    reranker = CVReranker()

    print("LLM provider check")
    print(f"  gemini configured: {bool(reranker.gemini_key)}")
    print(f"  groq configured  : {bool(reranker.groq_key)}")

    if not reranker.is_configured:
        print(
            "\nNo usable LLM key found.\n"
            "Set GEMINI_API_KEY or GROQ_API_KEY in .env. Values still left as\n"
            "'your_..._here' placeholders are treated as unset."
        )
        return 1

    results = {}
    if reranker.gemini_key:
        results["gemini"] = check_provider(reranker, "gemini")
    if reranker.groq_key:
        results["groq"] = check_provider(reranker, "groq")

    print("\n" + "=" * 60)
    for provider, ok in results.items():
        print(f"  {provider:8} {'OK' if ok else 'FAILED / unexpected ranking'}")

    if any(results.values()):
        print("\nAt least one provider works. You can now run:")
        print("  python -m evaluation.run_eval --ablations --rerank")
        return 0

    print("\nNo provider produced a usable ranking.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
