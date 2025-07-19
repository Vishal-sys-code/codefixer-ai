import faiss
import json
import logging
import numpy as np
import os
import google.generativeai as genai
import time
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """
    A class to retrieve documents from a FAISS index based on a query.
    """
    def __init__(self, index_path: str, metadata_path: str, google_api_key: str = None):
        """
        Initializes the Retriever with a FAISS index and metadata.

        Args:
            index_path: The path to the FAISS index file.
            metadata_path: The path to the JSONL metadata file.
        """
        self.index = faiss.read_index(index_path)
        self.metadata = []
        with open(metadata_path, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_AI_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key not provided. Please set the GOOGLE_AI_API_KEY environment variable.")
        genai.configure(api_key=self.google_api_key)

    def _embed(self, text: str) -> np.ndarray:
        """
        Embeds a string using Google's embedding-001 model.

        Args:
            text: The text to embed.

        Returns:
            A normalized NumPy array representing the embedding.
        """
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embedding = np.array(response["embedding"])
        return embedding / np.linalg.norm(embedding)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieves the top-k documents for a single query.

        Args:
            query: The query string.
            top_k: The number of documents to retrieve.

        Returns:
            A list of dictionaries, each containing a retrieved document.
        """
        start_time = time.time()
        query_embedding = self._embed(query)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            metadata = self.metadata[idx]
            results.append({
                "content": metadata["content"],
                "source": metadata["source"],
                "id": metadata["id"],
                "score": distances[0][i]
            })
        end_time = time.time()
        logger.info(f"Retrieval latency: {end_time - start_time:.4f} seconds")
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """
        Retrieves the top-k documents for a batch of queries.

        Args:
            queries: A list of query strings.
            top_k: The number of documents to retrieve for each query.

        Returns:
            A list of lists of dictionaries, where each inner list contains the retrieved documents for a query.
        """
        start_time = time.time()
        query_embeddings = np.array([self._embed(q) for q in queries])
        distances, indices = self.index.search(query_embeddings, top_k)
        batch_results = []
        for i, query_indices in enumerate(indices):
            results = []
            for j, idx in enumerate(query_indices):
                metadata = self.metadata[idx]
                results.append({
                    "content": metadata["content"],
                    "source": metadata["source"],
                    "id": metadata["id"],
                    "score": distances[i][j]
                })
            batch_results.append(results)
        end_time = time.time()
        logger.info(f"Batch retrieval latency for {len(queries)} queries: {end_time - start_time:.4f} seconds")
        return batch_results