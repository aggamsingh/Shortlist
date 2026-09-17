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
    chunk_cv,
    normalize_location,
    COMMON_CITIES,
)
from indexer.embedder import CVEmbedder

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

def word_boundary(term: str) -> str:
    """Regex matching `term` as a whole word."""
    return r"\b" + re.escape(term) + r"\b"

def extract_location(text: str) -> str:
    """Basic extraction of location based on common cities."""
    text_lower = text.lower()
    for city in COMMON_CITIES:
        # Match city with word boundaries
        if re.search(word_boundary(city), text_lower):
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
                )
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

    updated_files_count = 0
    skipped_files_count = 0

    # 6. Index each file
    for file_path in cv_files:
        filename = os.path.basename(file_path)
        try:
            file_hash = calculate_file_hash(file_path)
        except Exception:
            logger.error(f"Could not calculate hash for {filename}, skipping.")
            continue

        # Check if file has changed
        if file_path in state and state[file_path].get("hash") == file_hash:
            logger.debug(f"Skipping {filename} - hash unchanged.")
            skipped_files_count += 1
            continue

        logger.info(f"Indexing new or modified file: {filename}")
        
        # Parse CV text
        try:
            cv_text = parse_cv(file_path)
        except Exception as e:
            logger.error(f"Failed to parse CV text from {filename}: {e}. Skipping file.")
            continue

        if not cv_text.strip():
            logger.warning(f"Parsed CV text is empty for {filename}, skipping.")
            continue

        # Extract metadata
        candidate_name = clean_candidate_name(filename)
        years_exp = extract_years_of_experience(cv_text)
        location = extract_location(cv_text)
        
        # Generate deterministic UUIDs for candidate
        # Using filename and relative path helps keep candidate ID unique and persistent
        relative_path = os.path.relpath(file_path, CV_FOLDER_PATH)
        candidate_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, relative_path)
        
        logger.info(f"Candidate: '{candidate_name}' | ID: {candidate_uuid} | Exp: {years_exp} yrs | Loc: {location}")

        # Chunk CV text
        chunks = chunk_cv(cv_text)
        logger.info(f"Generated {len(chunks)} text chunks for {candidate_name}.")

        # Generate Embeddings
        try:
            embeddings = embedder.embed_texts(chunks)
        except Exception as e:
            logger.error(f"Failed to embed chunks for {candidate_name}: {e}. Skipping upsert.")
            continue

        # 7. Delete stale vectors for this candidate before inserting new ones
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
            logger.error(f"Failed to clear old vectors for {candidate_name}: {e}")

        # 8. Compile Qdrant Points and Upsert
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(candidate_uuid, f"chunk_{i}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "candidate_id": str(candidate_uuid),
                        "name": candidate_name,
                        # Normalise separators: os.path.join on Windows yields
                        # mixed "C:/a/b\c.docx", which looks broken in JSON.
                        "cv_path": file_path.replace(os.sep, "/"),
                        "chunk_text": chunk,
                        "years_of_experience": years_exp,
                        "location": location
                    }
                )
            )

        # Batch upsert points
        try:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )
            # Update index state
            state[file_path] = {
                "hash": file_hash,
                "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "candidate_name": candidate_name,
                "candidate_id": str(candidate_uuid)
            }
            updated_files_count += 1
            logger.info(f"Successfully indexed candidate {candidate_name}")
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant for {candidate_name}: {e}")

    # 9. Save State File
    save_index_state(STATE_FILE_PATH, state)
    logger.info(f"Pipeline complete. Indexed: {updated_files_count} | Unchanged: {skipped_files_count}")

if __name__ == "__main__":
    main()
