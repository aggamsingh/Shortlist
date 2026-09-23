import os
import re
from pypdf import PdfReader
from docx import Document
from indexer.utils import get_logger

logger = get_logger("indexer.parser")

COMMON_HEADERS = [
    "work experience", "experience", "professional experience", "employment history",
    "education", "academic background",
    "technical skills", "skills", "skills & expertise", "core competencies",
    "projects", "academic projects", "personal projects",
    "summary", "professional summary", "objective", "profile",
    "languages", "certifications", "achievements", "publications", "interests", "awards"
]

# Cities recognised for metadata extraction and query-side filtering.
# Aliases map to the single canonical spelling stored in the Qdrant payload.
CITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bangalore": "Bangalore",
    "gurugram": "Gurgaon",
    "gurgaon": "Gurgaon",
}
COMMON_CITIES = [
    "delhi", "mumbai", "bangalore", "bengaluru", "noida", "gurgaon",
    "gurugram", "pune", "hyderabad", "chennai", "kolkata"
]

def normalize_location(value: str) -> str:
    """Map any spelling of a known city to the canonical form stored in Qdrant.

    Used by the indexer when writing payloads and by the retriever when building
    filters, so a query for "Bengaluru" matches a CV indexed as "Bangalore".
    """
    if not value:
        return "Unknown"
    key = value.strip().lower()
    if key in CITY_ALIASES:
        return CITY_ALIASES[key]
    return value.strip().title()

def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text_parts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error parsing PDF file {file_path}: {e}")
        raise

def parse_docx(file_path: str) -> str:
    """Extract text from a DOCX file, including paragraphs and tables."""
    try:
        doc = Document(file_path)
        text_parts = []
        # Extract from paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        # Extract from tables (to capture work experiences inside tables)
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_parts.append(" | ".join(row_text))
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error parsing DOCX file {file_path}: {e}")
        raise

def parse_cv(file_path: str) -> str:
    """Determine CV file type and parse accordingly."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def extract_years_of_experience(text: str) -> int:
    """
    Attempt to extract years of experience from CV text using regex.
    Returns 0 if not found or cannot be parsed.
    """
    text_lower = text.lower()
    # Patterns like "5+ years", "4 years of experience", "experience: 3 years", "8 yrs", etc.
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience',
        r'experience\s*(?::|-)?\s*(\d+)\+?\s*(?:years?|yrs?)',
        r'worked\s+for\s+(\d+)\+?\s*(?:years?|yrs?)',
        r'total\s+experience\s*(?::|-)?\s*(\d+)\+?\s*(?:years?|yrs?)'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                # Return the maximum found value to be safe, capped at 40
                years = max(int(m) for m in matches)
                if years <= 40:
                    return years
            except ValueError:
                continue
    return 0

def clean_text(text: str) -> str:
    """Clean redundant spaces and newlines."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_by_words(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Helper to chunk text using word count windowing."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

def chunk_resume(text: str, strategy: str = None) -> list[str]:
    """Chunk a resume using the configured strategy.

    Default is the sliding window. That is an evidence-based choice, not an
    aesthetic one: on a corpus of realistic-length CVs with varied layouts, the
    window beats heading-based sectioning on both query sets (within-role nDCG@5
    0.790 vs 0.689, cross-role 0.932 vs 0.890) and yields perfect pool recall
    under hybrid retrieval.

    Heading-based sectioning looked reasonable in principle, and on the original
    55-word fixtures the two were indistinguishable. Once the fixtures grew to a
    realistic length the gap became consistent across every retrieval mode, so
    the default follows the measurement. Set CHUNK_STRATEGY=section to restore
    the previous behaviour.
    """
    strategy = (strategy or os.getenv("CHUNK_STRATEGY", "window")).strip().lower()
    if strategy == "section":
        return chunk_cv(text)
    return chunk_by_words(clean_text(text), chunk_size=200, overlap=50)


def chunk_cv(text: str) -> list[str]:
    """
    Intelligently split resume text into chunks.
    First tries section-based splitting, then falls back/sub-chunks.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # Identify headings
    sections = {}
    current_section = "Header"
    sections[current_section] = []
    
    for line in lines:
        is_header = False
        line_lower = line.lower().strip(":- ")
        if len(line_lower) < 40: # Headings are usually short
            for header in COMMON_HEADERS:
                if line_lower == header or line_lower.startswith(header + " "):
                    is_header = True
                    current_section = header
                    if current_section not in sections:
                        sections[current_section] = []
                    break
        
        if not is_header:
            sections[current_section].append(line)
            
    # Process sections
    chunks = []
    for sec_name, sec_lines in sections.items():
        sec_text = "\n".join(sec_lines)
        cleaned_sec = clean_text(sec_text)
        if not cleaned_sec:
            continue
            
        # Sub-chunk if section is too long
        word_count = len(cleaned_sec.split())
        if word_count > 250:
            sub_chunks = chunk_by_words(cleaned_sec, chunk_size=200, overlap=50)
            for sc in sub_chunks:
                chunks.append(f"[{sec_name.upper()}] {sc}")
        else:
            chunks.append(f"[{sec_name.upper()}] {cleaned_sec}")
            
    # If section-based chunking didn't yield much (e.g. poorly formatted CV)
    # fall back to standard sliding window over the full clean text
    if len(chunks) <= 1 or sum(len(c.split()) for c in chunks) < 30:
        cleaned_all = clean_text(text)
        return chunk_by_words(cleaned_all, chunk_size=200, overlap=50)
        
    return chunks
