
import os
import faiss
import numpy as np
import pickle
from langchain_openai import OpenAIEmbeddings
import config


class VectorStore:
    """
    Persistent FAISS-based vector store for long-term memory.
    Stores text chunks as embeddings and supports semantic similarity search.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY
        )
        self.index_path = config.FAISS_INDEX_PATH
        self.texts: list[str] = []
        self.index: faiss.IndexFlatL2 | None = None
        self._load()

    def _load(self):
        """Load existing FAISS index and texts from disk, or create fresh ones."""
        index_file = f"{self.index_path}.faiss"
        texts_file = f"{self.index_path}.pkl"

        if os.path.exists(index_file) and os.path.exists(texts_file):
            self.index = faiss.read_index(index_file)
            with open(texts_file, "rb") as f:
                self.texts = pickle.load(f)
            print(f"[VectorStore] Loaded {len(self.texts)} chunks from disk.")
        else:
            # Fresh index — dimension set on first add
            self.index = None
            self.texts = []
            print("[VectorStore] Starting fresh index.")

    def _save(self):
        """Persist the FAISS index and texts to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump(self.texts, f)

    def add_text(self, text: str):
        """
        Embed a text chunk and add it to the FAISS index.

        Args:
            text: Raw text string to store
        """
        if not text.strip():
            return

        # Split into chunks
        chunks = self._chunk_text(text)

        for chunk in chunks:
            embedding = self.embeddings.embed_query(chunk)
            vector = np.array([embedding], dtype=np.float32)

            # Initialise index on first entry
            if self.index is None:
                dim = len(embedding)
                self.index = faiss.IndexFlatL2(dim)

            self.index.add(vector)
            self.texts.append(chunk)

        self._save()
        print(f"[VectorStore] Added {len(chunks)} chunk(s). Total: {len(self.texts)}")

    def search(self, query: str, k: int = 3) -> list[str]:
        """
        Retrieve top-k most semantically similar chunks for a query.

        Args:
            query: Search string
            k: Number of results to return

        Returns:
            List of relevant text chunks
        """
        if self.index is None or len(self.texts) == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        vector = np.array([query_embedding], dtype=np.float32)

        k = min(k, len(self.texts))
        distances, indices = self.index.search(vector, k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.texts[idx])

        return results

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks for better retrieval.

        Args:
            text: Full text to chunk

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP

        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + config.CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]
