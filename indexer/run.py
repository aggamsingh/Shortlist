import os
import re
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import our custom modules
from indexer.utils import (
    get_logger,
    calculate_file_hash,
    connect_qdrant,
    load_index_state,
    save_index_state,
)
from indexer.parser import (
    parse_cv,
    extract_years_of_experience,
    chunk_resume,
    normalize_location,
    COMMON_CITIES,
)
from indexer.embedder import CVEmbedder
from indexer.sparse import BM25Encoder

# Load environment variables
load_dotenv()

logger = get_logger("indexer.run")

# Configuration variables
CV_FOLDER_PATH = os.getenv("CV_FOLDER_PATH", "./cvs")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "resumes")
STATE_FILE_PATH = os.getenv("STATE_FILE_PATH", "./data/index_state.json")
# When set, Qdrant runs embedded from this directory instead of over the network
# (no server, no Docker). Embedded mode holds an exclusive lock, so index first
# and start the API afterwards.
QDRANT_PATH = os.getenv("QDRANT_PATH")

# Hybrid (dense + BM25) retrieval. Costs a full re-parse of the corpus on every
# run, because BM25 needs corpus-wide statistics; set false to skip that and
# fall back to dense-only indexing.
HYBRID_RETRIEVAL = os.getenv("HYBRID_RETRIEVAL", "true").strip().lower() not in ("0", "false", "no")
BM25_STATE_PATH = os.getenv("BM25_STATE_PATH", "./data/bm25_state.json")

# The dense vector stays unnamed ("") so existing dense-only queries and any
# previously written points keep working; sparse is added as a named vector.
DENSE_VECTOR_NAME = ""
SPARSE_VECTOR_NAME = "text"

def word_boundary(term: str) -> str:
    """Regex matching `term` as a whole word."""
    return r"\b" + re.escape(term) + r"\b"

# Lines at the top of a CV that plausibly hold contact details. A candidate's
# own location lives here; cities further down usually belong to an employer or
# a university.
CONTACT_BLOCK_LINES = 6


def extract_location(text: str) -> str:
    """Extract the candidate's city, preferring the contact block at the top.

    Scanning the whole document and taking the first match is wrong on real CVs:
    an education section reading "University of Delhi" or an employer address
    will outvote the candidate's actual city. Found when the evaluation corpus
    grew realistic education entries and ten candidates suddenly relocated.

    The header is searched first; the rest of the document is only a fallback,
    which still beats returning Unknown for a CV that lists its city late.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header = "\n".join(lines[:CONTACT_BLOCK_LINES]).lower()
    body = text.lower()

    for scope in (header, body):
        for city in COMMON_CITIES:
            if re.search(word_boundary(city), scope):
                return normalize_location(city)
    return "Unknown"

def clean_candidate_name(filename: str) -> str:
    """Generate a clean candidate name from the file name."""
    name_part = os.path.splitext(filename)[0]
    # Replace underscores and hyphens with spaces first so we get individual tokens
    name_part = re.sub(r'[_\-]', ' ', name_part)
    # Remove year-like numbers (4-digit sequences)
    name_part = re.sub(r'\b\d{4}\b', '', name_part)
    # Remove remaining standalone digits
    name_part = re.sub(r'\b\d+\b', '', name_part)
    # Remove common resume/CV keywords (case-insensitive, as whole words)
    name_part = re.sub(r'(?i)\b(?:resume|cv|profile|doc|pdf|latest|updated)\b', '', name_part)
    # Strip any remaining non-alpha characters
    name_part = re.sub(r'[^a-zA-Z\s]', ' ', name_part)
    # Collapse multiple spaces
    name_part = re.sub(r'\s+', ' ', name_part).strip()

    if not name_part:
        return "Unknown Candidate"
    return name_part.title()

# Payload fields that must be indexed in Qdrant.
#   candidate_id -> grouped retrieval (query_points_groups) and stale-vector deletes
#   location / years_of_experience -> screening filters
# Without these, Qdrant still answers correctly but falls back to a full payload
# scan, which does not hold up at a few hundred thousand chunks.
PAYLOAD_INDEXES = {
    "candidate_id": models.PayloadSchemaType.KEYWORD,
    "location": models.PayloadSchemaType.KEYWORD,
    "years_of_experience": models.PayloadSchemaType.INTEGER,
}


def ensure_payload_indexes(qdrant_client) -> None:
    """Create payload indexes if absent. Safe to call on every run."""
    for field_name, schema in PAYLOAD_INDEXES.items():
        try:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=schema,
            )
            logger.info(f"Created payload index on '{field_name}'.")
        except Exception as e:
            # Qdrant returns an error when the index already exists; that is the
            # normal path on re-runs and must not abort indexing.
            logger.debug(f"Payload index on '{field_name}' not created ({e}); assuming it exists.")


def main():
    logger.info("Starting CV Indexing pipeline...")
    
    # 1. Connect to Qdrant
    try:
        if QDRANT_PATH:
            qdrant_client = QdrantClient(path=QDRANT_PATH)
            logger.info(f"Opened embedded Qdrant at {QDRANT_PATH}")
        else:
            qdrant_client = connect_qdrant(
                QDRANT_HOST, QDRANT_PORT,
                client_factory=lambda: QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT),
            )
            logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        logger.critical(f"Failed to connect to Qdrant: {e}")
        return

    # 2. Instantiate Embedder to get dimension size
    try:
        embedder = CVEmbedder()
        vector_dim = embedder.dimension
    except Exception as e:
        logger.critical(f"Failed to initialize embedding model: {e}")
        return

    # 3. Ensure Collection Exists in Qdrant
    try:
        exists = qdrant_client.collection_exists(QDRANT_COLLECTION)
        if not exists:
            logger.info(f"Collection '{QDRANT_COLLECTION}' not found. Creating new collection.")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=vector_dim,
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
            )
            logger.info(f"Collection '{QDRANT_COLLECTION}' created successfully.")
        else:
            logger.info(f"Collection '{QDRANT_COLLECTION}' already exists.")
        ensure_payload_indexes(qdrant_client)
    except Exception as e:
        logger.critical(f"Error checking/creating Qdrant collection: {e}")
        return

    # 4. Load Index State
    state = load_index_state(STATE_FILE_PATH)
    logger.info(f"Loaded index state. Tracks {len(state)} files.")

    # 5. Scan files in CV_FOLDER_PATH
    if not os.path.exists(CV_FOLDER_PATH):
        logger.warning(f"CV Folder path '{CV_FOLDER_PATH}' does not exist on host. Creating it.")
        os.makedirs(CV_FOLDER_PATH, exist_ok=True)

    cv_files = []
    for root, _, files in os.walk(CV_FOLDER_PATH):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in [".pdf", ".docx"]:
                cv_files.append(os.path.join(root, file))
            else:
                logger.warning(f"Skipping file {file} with unsupported extension '{ext}'")

    logger.info(f"Found {len(cv_files)} candidate CV files to check.")

    # 6. Pass one: parse and chunk everything.
    #
    # BM25 needs corpus-wide statistics (document frequency, average length), so
    # every CV has to be read even when only one has changed. Parsing is cheap
    # next to embedding -- the expensive step still runs only for changed files
    # in pass two. With HYBRID_RETRIEVAL off, unchanged files are skipped here
    # entirely and this becomes the original single-pass behaviour.
    parsed = {}
    for file_path in cv_files:
        filename = os.path.basename(file_path)
        try:
            file_hash = calculate_file_hash(file_path)
        except Exception:
            logger.error(f"Could not calculate hash for {filename}, skipping.")
            continue

        unchanged = file_path in state and state[file_path].get("hash") == file_hash
        if unchanged and not HYBRID_RETRIEVAL:
            logger.debug(f"Skipping {filename} - hash unchanged.")
            continue

        try:
            cv_text = parse_cv(file_path)
        except Exception as e:
            logger.error(f"Failed to parse CV text from {filename}: {e}. Skipping file.")
            continue

        if not cv_text.strip():
            logger.warning(f"Parsed CV text is empty for {filename}, skipping.")
            continue

        relative_path = os.path.relpath(file_path, CV_FOLDER_PATH)
        parsed[file_path] = {
            "hash": file_hash,
            "unchanged": unchanged,
            "name": clean_candidate_name(filename),
            "years": extract_years_of_experience(cv_text),
            "location": extract_location(cv_text),
            "candidate_uuid": uuid.uuid5(uuid.NAMESPACE_DNS, relative_path),
            "chunks": chunk_resume(cv_text),
        }

    # 7. Fit BM25 across every chunk in the corpus.
    encoder = BM25Encoder()
    if HYBRID_RETRIEVAL and parsed:
        encoder.fit(chunk for doc in parsed.values() for chunk in doc["chunks"])

    def sparse_vector(text: str):
        indices, values = encoder.encode_document(text)
        return models.SparseVector(indices=indices, values=values)

    # 8. Pass two: embed and upsert changed files; refresh sparse vectors on the rest.
    updated_files_count = 0
    refreshed_files_count = 0
    skipped_files_count = 0

    for file_path, doc in parsed.items():
        candidate_uuid = doc["candidate_uuid"]
        chunks = doc["chunks"]
        point_ids = [str(uuid.uuid5(candidate_uuid, f"chunk_{i}")) for i in range(len(chunks))]

        if doc["unchanged"]:
            # Document weights depend on average corpus length, which shifts as
            # CVs are added. Refreshing the sparse side keeps BM25 consistent
            # without paying to re-embed; update_vectors leaves the dense vector
            # untouched.
            if HYBRID_RETRIEVAL:
                try:
                    qdrant_client.update_vectors(
                        collection_name=QDRANT_COLLECTION,
                        points=[
                            models.PointVectors(id=pid, vector={SPARSE_VECTOR_NAME: sparse_vector(chunk)})
                            for pid, chunk in zip(point_ids, chunks)
                        ],
                    )
                    refreshed_files_count += 1
                except Exception as e:
                    logger.warning(f"Could not refresh sparse vectors for {doc['name']}: {e}")
            skipped_files_count += 1
            continue

        logger.info(f"Indexing new or modified file: {os.path.basename(file_path)}")
        logger.info(
            f"Candidate: '{doc['name']}' | ID: {candidate_uuid} | "
            f"Exp: {doc['years']} yrs | Loc: {doc['location']}"
        )
        logger.info(f"Generated {len(chunks)} text chunks for {doc['name']}.")

        try:
            embeddings = embedder.embed_texts(chunks)
        except Exception as e:
            logger.error(f"Failed to embed chunks for {doc['name']}: {e}. Skipping upsert.")
            continue

        # Clear stale vectors for this candidate before inserting new ones.
        try:
            qdrant_client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="candidate_id",
                            match=models.MatchValue(value=str(candidate_uuid))
                        )
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Failed to clear old vectors for {doc['name']}: {e}")

        points = []
        for point_id, chunk, dense in zip(point_ids, chunks, embeddings):
            vector = {DENSE_VECTOR_NAME: dense}
            if HYBRID_RETRIEVAL:
                vector[SPARSE_VECTOR_NAME] = sparse_vector(chunk)
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "candidate_id": str(candidate_uuid),
                        "name": doc["name"],
                        # Normalise separators: os.path.join on Windows yields
                        # mixed "C:/a/b\c.docx", which looks broken in JSON.
                        "cv_path": file_path.replace(os.sep, "/"),
                        "chunk_text": chunk,
                        "years_of_experience": doc["years"],
                        "location": doc["location"],
                    },
                )
            )

        try:
            qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            state[file_path] = {
                "hash": doc["hash"],
                "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "candidate_name": doc["name"],
                "candidate_id": str(candidate_uuid),
            }
            updated_files_count += 1
            logger.info(f"Successfully indexed candidate {doc['name']}")
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant for {doc['name']}: {e}")

    # 9. Persist state. The BM25 statistics live next to the index because the
    # API needs the identical IDF values to encode queries.
    save_index_state(STATE_FILE_PATH, state)
    if HYBRID_RETRIEVAL and encoder.is_fitted:
        encoder.save(BM25_STATE_PATH)

    logger.info(
        f"Pipeline complete. Indexed: {updated_files_count} | "
        f"Unchanged: {skipped_files_count} (sparse refreshed: {refreshed_files_count})"
    )

if __name__ == "__main__":
    main()
