# Retrieval-Augmented Code Debugger `codefixer-ai`

This project is a retrieval-augmented code debugger that uses a large language model (LLM) to help developers debug their code. It takes an error message and a code snippet as input, retrieves relevant context from a knowledge base of GitHub issues and Stack Overflow questions, and then uses the LLM to generate a patch and a unit test to fix the bug.

## Components

The project is composed of the following components:

- **`app.py`**: A Streamlit web application that provides a user interface for the debugger.
- **`indexer.py`**: A script that builds a FAISS index of GitHub issues and Stack Overflow questions.
- **`retriever.py`**: A class that retrieves relevant documents from the FAISS index.
- **`llm_agent.py`**: A class that uses a large language model to generate a patch and a unit test.
- **`patch_parser.py`**: A script that parses the output of the LLM to extract the patch and the unit test.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/retrieval-augmented-code-debugger.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```bash
   cp .env.example .env
   ```
   Then, edit the `.env` file to add your GitHub and Google API keys.

## Usage

1. Build the FAISS index:
   ```bash
   python indexer.py --repo_name <repo-name> --so_tags <so-tags>
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open the application in your browser and paste your error message and code snippet into the text area.
4. Click the "Run Debugger" button to generate a patch and a unit test.
