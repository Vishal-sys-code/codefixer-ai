import os
import json
import faiss
import numpy as np

def create_dummy_index():
    # Create a dummy FAISS index
    index_path = "data/faiss_index/index.faiss"
    metadata_path = "data/github_issues.jsonl"
    d = 128  # dimension of the embeddings
    index = faiss.IndexFlatL2(d)
    embeddings = np.random.rand(10, d).astype('float32')
    index.add(embeddings)
    faiss.write_index(index, index_path)

    # Create dummy metadata
    metadata = []
    for i in range(10):
        metadata.append({
            "text": f"This is document {i}",
            "source": "test",
            "id": i
        })
    with open(metadata_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

if __name__ == '__main__':
    create_dummy_index()