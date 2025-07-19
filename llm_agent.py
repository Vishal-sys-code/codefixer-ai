import os
import logging
import google.generativeai as genai
import time
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided. Please set the GOOGLE_API_KEY environment variable.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_patch(self, context: str, file_path: str = None, retries: int = 3, delay: int = 60):
        system_prompt = """You are an expert programmer. Your task is to provide code patches to fix bugs.
Pay close attention to data types and potential `TypeError` exceptions.
Analyze the provided context, which includes an error message and relevant code snippets.
Generate a patch in the git diff format.
Also, generate a relevant unit test to verify the fix.

Your response should be in the following format:

**Patch:**
```diff
--- a/path/to/file
+++ b/path/to/file
@@ -1,1 +1,1 @@
- old code
+ new code
```

**Unit Test:**
```python
import unittest

class TestMyCode(unittest.TestCase):
    def test_my_function(self):
        self.assertEqual(my_function(1), 1)
```
"""
        
        if file_path:
            user_prompt = f"Given the following context and file path, generate a patch to fix the bug.\n\nContext:\n{context}\n\nFile Path:\n{file_path}"
        else:
            user_prompt = f"Given the following context, generate a patch to fix the bug.\n\nContext:\n{context}"

        prompt = f"{system_prompt}\n\n{user_prompt}"

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        for i in range(retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "text/plain"},
                    safety_settings=safety_settings
                )

                logger.info(f"Full API Response: {response}")

                if hasattr(response, 'text'):
                    return response.text
                
                # Check for blocked response
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.error(f"API call blocked due to: {response.prompt_feedback.block_reason}")
                    return f"Error: The API call was blocked. Reason: {response.prompt_feedback.block_reason}"

                # Handle other unexpected responses
                logger.error(f"Unexpected API response: {response}")
                return "Error: The API returned an unexpected response. Check logs for details."

            except Exception as e:
                if "429" in str(e) and i < retries - 1:
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"An error occurred during the API call: {e}")
                    return f"Error: An error occurred during the API call: {e}"