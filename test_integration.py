import unittest
from unittest.mock import patch, MagicMock
import os
import json
import faiss
import numpy as np
from retriever import Retriever
from llm_agent import LLMAgent
from patch_parser import parse_llm_output

class TestIntegration(unittest.TestCase):
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

    @patch('retriever.genai.embed_content')
    @patch('llm_agent.genai.GenerativeModel')
    def test_rag_pipeline(self, mock_generative_model, mock_embed_content):
        # Mock the embedding function
        mock_embed_content.return_value = {"embedding": np.random.rand(self.d).astype('float32')}

        # Mock the LLM
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value.text = "This is a test patch."
        mock_generative_model.return_value = mock_model_instance

        # 1. Instantiate components
        retriever = Retriever(
            index_path=self.index_path,
            metadata_path=self.metadata_path,
            google_api_key="fake_key"
        )
        llm_agent = LLMAgent(api_key="fake_key")

        # 2. Retrieve context
        retrieved_docs = retriever.retrieve("test query", top_k=3)
        self.assertEqual(len(retrieved_docs), 3)

        # 3. Generate patch
        context_str = "\\n".join([doc['text'] for doc in retrieved_docs])
        full_prompt = f"Error and Code:\\n'test error'\\n\\nRetrieved Context:\\n{context_str}"
        llm_response = llm_agent.generate_patch(full_prompt)

        # 4. Parse and display output
        parsed_output = parse_llm_output(llm_response)
        
        self.assertIn("patches", parsed_output)
        self.assertIn("unit_tests", parsed_output)
        self.assertEqual(len(parsed_output["patches"]), 0)
        self.assertEqual(len(parsed_output["unit_tests"]), 0)


if __name__ == '__main__':
    unittest.main()
