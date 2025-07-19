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
        # Get the actual call arguments
        actual_call_args = mock_model_instance.generate_content.call_args
        # Check if the prompt contains the expected text
        self.assertIn("test prompt", actual_call_args[0][0])


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
        
        # Act
        response = agent.generate_patch("test prompt")
        
        # Assert
        self.assertTrue(response.startswith("Error: An error occurred during the API call:"))

if __name__ == '__main__':
    unittest.main()
