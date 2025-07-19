import unittest
from unittest.mock import patch, MagicMock
from llm_agent import LLMAgent

class TestLLMAgent(unittest.TestCase):

    @patch('llm_agent.genai.GenerativeModel')
    def test_generate_patch_success(self, mock_generative_model):
        # Arrange
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value.text = "This is a test patch."
        mock_generative_model.return_value = mock_model_instance
        
        agent = LLMAgent(api_key="fake_api_key")
        
        # Act
        response = agent.generate_patch("test prompt")
        
        # Assert
        self.assertEqual(response, "This is a test patch.")
        mock_generative_model.assert_called_once()
        mock_model_instance.generate_content.assert_called_once_with(
            "Given the following context, generate a patch to fix the bug.\n\nContext:\ntest prompt",
            generation_config={"response_mime_type": "text/plain"}
        )

    @patch('llm_agent.genai.GenerativeModel')
    def test_generate_patch_with_file_path(self, mock_generative_model):
        # Arrange
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value.text = "This is a test patch for a file."
        mock_generative_model.return_value = mock_model_instance
        
        agent = LLMAgent(api_key="fake_api_key")
        
        # Act
        response = agent.generate_patch("test prompt", file_path="/path/to/file.py")
        
        # Assert
        self.assertEqual(response, "This is a test patch for a file.")
        mock_model_instance.generate_content.assert_called_once()
        self.assertIn("file.py", mock_model_instance.generate_content.call_args[0][0])


    @patch('llm_agent.genai.GenerativeModel')
    def test_generate_patch_api_error(self, mock_generative_model):
        # Arrange
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        mock_generative_model.return_value = mock_model_instance
        
        agent = LLMAgent(api_key="fake_api_key")
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            agent.generate_patch("test prompt")
        
        self.assertTrue('API Error' in str(context.exception))

if __name__ == '__main__':
    unittest.main()
