import os
import logging
import google.generativeai as genai

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

    def generate_patch(self, context: str, file_path: str = None):
        if file_path:
            prompt = f"Given the following context and file path, generate a patch to fix the bug.\n\nContext:\n{context}\n\nFile Path:\n{file_path}"
        else:
            prompt = f"Given the following context, generate a patch to fix the bug.\n\nContext:\n{context}"

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

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "text/plain"},
                safety_settings=safety_settings
            )

            if hasattr(response, 'text'):
                return response.text
            else:
                logger.error(f"Unexpected API response: {response}")
                return "Error: The API returned an unexpected response."

        except Exception as e:
            logger.error(f"An error occurred during the API call: {e}")
            return "Error: An error occurred during the API call."