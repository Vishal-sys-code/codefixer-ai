import unittest
import unittest.mock
import os
import json
import faiss
import numpy as np
from retriever import Retriever

class TestRetriever(unittest.TestCase):
    def setUp(self):
        # Create a dummy FAISS index
        self.index_path = "test_index.faiss"
        self.metadata_path = "test_metadata.jsonl"
        self.d = 128  # dimension of the embeddings
        self.index = faiss.IndexFlatL2(self.d)
        self.embeddings = np.random.rand(10, self.d).astype('float32')
        self.index.add(self.embeddings)
        faiss.write_index(self.index, self.index_path)

        # Create dummy metadata
        self.metadata = []
        for i in range(10):
            self.metadata.append({
                "text": f"This is document {i}",
                "source": "test",
                "id": i
            })
        with open(self.metadata_path, "w") as f:
            for item in self.metadata:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        os.remove(self.index_path)
        os.remove(self.metadata_path)

    @unittest.mock.patch('retriever.genai.embed_content')
    def test_retriever(self, mock_embed_content):
        # We can't fully test the retriever without a valid API key.
        # We will create a mock for the genai.embed_content function
        # to simulate the API call and test the rest of the logic.
        mock_embed_content.return_value = {"embedding": np.random.rand(self.d).astype('float32')}
        retriever = Retriever(self.index_path, self.metadata_path, google_api_key="fake_key")
        results = retriever.retrieve("test query", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertTrue("text" in results[0])
        self.assertTrue("source" in results[0])
        self.assertTrue("id" in results[0])
        self.assertTrue("score" in results[0])

if __name__ == '__main__':
    unittest.main()