import faiss
import github
import logging
import numpy as np
import google.generativeai as genai
import os
import re
import stackapi

from typing import Dict, List


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self, repo_name: str, so_tags: List[str], github_token: str = None, google_api_key: str = None):
        self.repo_name = repo_name
        self.so_tags = so_tags
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.index = None
        self.metadata = []

        if not self.github_token:
            raise ValueError("GitHub token not provided. Please set the GITHUB_TOKEN environment variable.")
        if not self.google_api_key:
            raise ValueError("Google API key not provided. Please set the GOOGLE_API_KEY environment variable.")

        self.gh = github.Github(self.github_token)
        self.so = stackapi.StackAPI("stackoverflow")
        genai.configure(api_key=self.google_api_key)

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_github_issues(self):
        logger.info(f"Fetching open issues from {self.repo_name}...")
        repo = self.gh.get_repo(self.repo_name)
        issues = repo.get_issues(state="open")
        documents = []
        for issue in issues:
            comments = "".join([c.body for c in issue.get_comments()])
            full_text = f"{issue.title} {issue.body} {comments}"
            processed_text = self._preprocess_text(full_text)
            documents.append({
                "source": "github",
                "url": issue.html_url,
                "id": issue.id,
                "document": processed_text,
            })
        logger.info(f"Fetched {len(documents)} issues from GitHub.")
        return documents

    def _get_stackoverflow_questions(self):
        logger.info(f"Fetching questions from Stack Overflow with tags: {self.so_tags}...")
        questions = self.so.fetch("questions", tagged=self.so_tags, sort="votes", pagesize=100, filter="withbody")
        documents = []
        for q in questions["items"]:
            if q.get("is_answered") and q.get("accepted_answer_id"):
                answer = self.so.fetch(f"answers/{q['accepted_answer_id']}", filter="withbody")["items"][0]
                full_text = f"{q['title']} {q['body']} {answer['body']}"
                processed_text = self._preprocess_text(full_text)
                documents.append({
                    "source": "stackoverflow",
                    "url": q["link"],
                    "id": q["question_id"],
                    "document": processed_text,
                })
        logger.info(f"Fetched {len(documents)} questions from Stack Overflow.")
        return documents

    def build_index(self):
        documents = self._get_github_issues()
        documents.extend(self._get_stackoverflow_questions())

        logger.info("Generating embeddings...")
        embeddings = []
        for doc in documents:
            response = genai.embed_content(
                model="models/embedding-001",
                content=doc["document"],
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.append(response["embedding"])
        
        embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = documents
        logger.info("Index built successfully.")

    def save_index(self, path: str):
        logger.info(f"Saving index to {path}...")
        if not os.path.exists(path):
            os.makedirs(path)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.json"), "w") as f:
            import json
            json.dump(self.metadata, f)
        logger.info("Index saved successfully.")

    def load_index(self, path: str):
        logger.info(f"Loading index from {path}...")
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.json"), "r") as f:
            import json
            self.metadata = json.load(f)
        logger.info("Index loaded successfully.")

    def query_index(self, query: str, top_k: int) -> List[Dict]:
        logger.info(f"Querying index with top_k={top_k}...")
        response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array(response["embedding"]).astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "score": distances[0][i],
                "metadata": self.metadata[idx]
            })
        logger.info("Query processed successfully.")
        return results