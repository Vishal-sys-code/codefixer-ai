import os
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_PATH = os.getenv("INDEX_PATH", "data/faiss_index")
METADATA_PATH = os.getenv("METADATA_PATH", "data/github_issues.jsonl")

from retriever import Retriever
from llm_agent import LLMAgent
from patch_parser import parse_llm_output

@patch('retriever.genai.embed_content')
@patch('llm_agent.genai.GenerativeModel')
def run_debug_logic(mock_generative_model, mock_embed_content, error_snippet, repo_path):
    """
    This function simulates the core logic of the Streamlit app's debugger.
    """
    # Mock the embedding function
    mock_embed_content.return_value = {"embedding": [0.1] * 128}

    # Mock the LLM
    mock_model_instance = MagicMock()
    mock_model_instance.generate_content.return_value.text = "This is a test patch."
    mock_generative_model.return_value = mock_model_instance

    try:
        # 1. Instantiate components
        retriever = Retriever(
            index_path=f"{INDEX_PATH}/index.faiss",
            metadata_path=METADATA_PATH,
            google_api_key="fake_key"
        )
        llm_agent = LLMAgent(api_key="fake_key")

        # 2. Retrieve context
        retrieved_docs = retriever.retrieve(error_snippet, top_k=5)

        # 3. Generate patch
        context_str = "\n".join([doc['text'] for doc in retrieved_docs])
        full_prompt = f"Error and Code:\n{error_snippet}\n\nRetrieved Context:\n{context_str}"
        llm_response = llm_agent.generate_patch(full_prompt, file_path=repo_path)

        # 4. Parse and display output
        parsed_output = parse_llm_output(llm_response)

        return parsed_output

    except FileNotFoundError:
        return {"error": f"Index not found at {INDEX_PATH}. Please run the indexer first."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

if __name__ == "__main__":
    # Simulate the inputs from the Streamlit app
    with open("syntax_error.py", "r") as f:
        error_snippet = f.read()
    
    repo_path = os.getcwd()

    # Run the debugging logic
    result = run_debug_logic(error_snippet=error_snippet, repo_path=repo_path)

    # Print the results
    print("--- Debugging Results ---")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Suggested Patch:")
        if result["patches"]:
            for patch in result["patches"]:
                print(patch['diff'])
        else:
            print("No patch was generated.")

        print("\nGenerated Unit Tests:")
        if result["unit_tests"]:
            for test in result["unit_tests"]:
                print(test['code'])
        else:
            print("No unit tests were generated.")