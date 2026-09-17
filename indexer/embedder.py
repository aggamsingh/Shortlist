from sentence_transformers import SentenceTransformer
from indexer.utils import get_logger

logger = get_logger("indexer.embedder")

class CVEmbedder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CVEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if self._initialized:
            return
        logger.info(f"Initializing CVEmbedder with model: {model_name} on CPU")
        
        try:
            # Force CPU execution
            self.model = SentenceTransformer(model_name, device="cpu")
            # Support both old and new API for embedding dimension
            if hasattr(self.model, 'get_embedding_dimension'):
                self.dimension = self.model.get_embedding_dimension()
            else:
                self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            raise
        self._initialized = True

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        return embeddings.tolist()
