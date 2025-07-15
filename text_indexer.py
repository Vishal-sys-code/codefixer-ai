import os
from indexer import Indexer

# This is a basic test script to verify the functionality of the Indexer class.
# To run this script, you need to set the following environment variables:
# export GITHUB_TOKEN="your_github_token"
# export GOOGLE_API_KEY="your_google_api_key"

def main():
    repo_name = "google/generative-ai-python"
    so_tags = ["python", "error-handling"]
    
    indexer = Indexer(repo_name=repo_name, so_tags=so_tags)
    
    # Build the index
    indexer.build_index()
    
    # Save the index
    indexer.save_index("my_index")
    
    # Load the index
    loaded_indexer = Indexer(repo_name=repo_name, so_tags=so_tags)
    loaded_indexer.load_index("my_index")
    
    # Query the index
    query = "How to handle API errors?"
    results = loaded_indexer.query_index(query, top_k=5)
    
    print(f"Query: {query}")
    print("Results:")
    for result in results:
        print(f"  Score: {result['score']}")
        print(f"  Source: {result['metadata']['source']}")
        print(f"  URL: {result['metadata']['url']}")
        print(f"  Document: {result['metadata']['document'][:200]}...")
        print("-" * 20)

if __name__ == "__main__":
    main()