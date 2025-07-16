import os
import json
import time
import re
from typing import Dict, List, Iterator

from github import Github
from stackapi import StackAPI
from bs4 import BeautifulSoup


class DataCollector:
    """
    A class to collect data from GitHub issues and StackOverflow posts,
    clean it, and save it to JSONL files.
    """

    def __init__(self, github_repo: str, so_tags: List[str], github_token: str = None):
        """
        Initializes the DataCollector.

        Args:
            github_repo: The GitHub repository in the format "owner/repo_name".
            so_tags: A list of StackOverflow tags.
            github_token: A GitHub personal access token for higher rate limits.
        """
        self.github_repo = github_repo
        self.so_tags = so_tags
        self.github_api = Github(github_token)
        self.stackoverflow_api = StackAPI('stackoverflow')

    def clean_text(self, raw_text: str) -> str:
        """
        Cleans the input text by removing markdown code fences, HTML tags,
        and normalizing whitespace.
        """
        if not raw_text:
            return ""
        # Remove markdown code fences
        text = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL)
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        # Normalize whitespace
        text = " ".join(text.split())
        return text

    def fetch_github_issues(self) -> Iterator[Dict]:
        """
        Fetches all issues (open and closed) from the target GitHub repository.
        """
        repo = self.github_api.get_repo(self.github_repo)
        issues = repo.get_issues(state='all')
        for issue in issues:
            try:
                comments = " ".join([comment.body for comment in issue.get_comments()])
                yield {
                    "id": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                    "comments": comments,
                    "source": "github",
                    "repo_or_tag": self.github_repo,
                }
            except Exception as e:
                print(f"Error fetching issue {issue.number}: {e}")
                time.sleep(60)  # Wait a minute before retrying

    def fetch_stackoverflow(self) -> Iterator[Dict]:
        """
        Fetches the top 5000 questions from StackOverflow with the given tags.
        """
        questions = self.stackoverflow_api.fetch('questions', tagged=self.so_tags, sort='votes', pagesize=100, max_pages=50)
        for question in questions['items']:
            if question.get('is_answered', False) and 'accepted_answer_id' in question:
                accepted_answer = self.stackoverflow_api.fetch(f"answers/{question['accepted_answer_id']}", filter='withbody')
                if accepted_answer['items']:
                    yield {
                        "id": question['question_id'],
                        "title": question['title'],
                        "body": question['body'],
                        "accepted_answer": accepted_answer['items'][0]['body'],
                        "source": "stackoverflow",
                        "repo_or_tag": ",".join(self.so_tags),
                    }

    def run(self):
        """
        Runs the data collection and preprocessing pipeline.
        """
        os.makedirs("data", exist_ok=True)

        print("Fetching openai-python issues...")
        with open("data/openai_issues.jsonl", "w") as f:
            for issue in self.fetch_github_issues():
                content = f"{issue['title']} {issue['body']} {issue['comments']}"
                cleaned_content = self.clean_text(content)
                record = {
                    "id": issue['id'],
                    "content": cleaned_content,
                    "source": issue['source'],
                    "repo_or_tag": issue['repo_or_tag'],
                }
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Example usage:
    # You will need to create a GitHub personal access token and set it as an
    # environment variable named GITHUB_TOKEN
    github_token = os.environ.get("GITHUB_TOKEN")
    collector = DataCollector(
        github_repo="openai/openai-python",
        so_tags=["python", "error-handling"],
        github_token=github_token,
    )
    collector.run()