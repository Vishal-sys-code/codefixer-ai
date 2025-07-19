import streamlit as st
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_PATH = os.getenv("INDEX_PATH", "data/faiss_index")
METADATA_PATH = os.getenv("METADATA_PATH", "data/github_issues.jsonl")

from indexer import Indexer
from retriever import Retriever
from llm_agent import LLMAgent
from patch_parser import parse_llm_output

def main():
    st.set_page_config(layout="wide")
    st.title("Retrieval-Augmented Code Debugger")

    # --- Sidebar for Inputs ---
    st.sidebar.header("Debugger Inputs")
    error_snippet = st.sidebar.text_area(
        "Error & Code Snippet",
        height=300,
        placeholder="Paste your stack trace and relevant code here..."
    )
    repo_path = st.sidebar.text_input(
        "Repository Path (Optional)",
        placeholder="e.g., /path/to/your/repo"
    )
    debug_button = st.sidebar.button("Run Debugger")

    # --- Main Area for Outputs ---
    st.header("Debugging Results")

    if debug_button:
        if not error_snippet:
            st.warning("Please provide an error and code snippet.")
        else:
            with st.spinner("Running debugger..."):
                try:
                    # 1. Instantiate components
                    # Note: Indexer might not be needed if an index already exists.
                    # For this example, we assume a pre-built index.
                    retriever = Retriever(
                        index_path=f"{INDEX_PATH}/index.faiss",
                        metadata_path=METADATA_PATH,
                        google_api_key=GOOGLE_API_KEY
                    )
                    llm_agent = LLMAgent(api_key=GOOGLE_API_KEY)

                    # 2. Retrieve context
                    retrieved_docs = retriever.retrieve(error_snippet, top_k=5)
                    
                    with st.expander("Retrieved Context"):
                        for doc in retrieved_docs:
                            st.write(f"**Source:** {doc['source']} ({doc['id']}) - **Score:** {doc['score']:.4f}")
                            st.text(doc['content'])
                            st.divider()

                    # 3. Generate patch
                    context_str = "\n".join([doc['content'] for doc in retrieved_docs])
                    full_prompt = f"Error and Code:\n{error_snippet}\n\nRetrieved Context:\n{context_str}"
                    llm_response = llm_agent.generate_patch(full_prompt, file_path=repo_path)

                    # 4. Parse and display output
                    parsed_output = parse_llm_output(llm_response)
                    
                    st.subheader("Suggested Patch")
                    if parsed_output["patches"]:
                        for patch in parsed_output["patches"]:
                            st.code(patch['diff'], language='diff')
                    else:
                        st.write("No patch was generated.")

                    st.subheader("Generated Unit Tests")
                    if parsed_output["unit_tests"]:
                        for test in parsed_output["unit_tests"]:
                            st.code(test['code'], language='python')
                    else:
                        st.write("No unit tests were generated.")

                    st.success("Debugging complete!")
                
                except FileNotFoundError:
                    st.error(f"Index not found at {INDEX_PATH}. Please run the indexer first.")
                except Exception as e:
                    st.error("An error occurred while processing your request.")
                    st.error(f"Error details: {e}")
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()