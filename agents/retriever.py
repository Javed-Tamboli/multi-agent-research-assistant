
from memory.vector_store import VectorStore
import config


class RetrieverAgent:
    """
    Retrieves relevant past research from the FAISS vector store
    based on semantic similarity to the current query.
    """

    def __init__(self):
        self.vector_store = VectorStore()

    def retrieve(self, query: str) -> list[str]:
        """
        Returns top-k relevant document chunks from memory.

        Args:
            query: The user's research question

        Returns:
            List of relevant text chunks
        """
        results = self.vector_store.search(query, k=config.MAX_RETRIEVAL_RESULTS)

        if not results:
            print("[Retriever] No relevant memory found.")
            return []

        print(f"[Retriever] Found {len(results)} relevant memory chunks.")
        return results
