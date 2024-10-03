import unittest
from unittest.mock import patch, MagicMock
from engineer import StructuralEngineerAgent

class TestStructuralEngineerAgent(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        self.agent = StructuralEngineerAgent()

    @patch('engineer.OpenAI')
    @patch('engineer.FAISS')
    @patch('engineer.OpenAIEmbeddings')
    def test_handle_validate_request(self, mock_embeddings, mock_faiss, mock_openai):
        # Mock the OpenAI client response
        mock_openai.return_value.chat.completions.create.return_value.choices[0].message.content = "This is a standard procedure."

        # Mock the FAISS similarity search
        mock_faiss.return_value.similarity_search.return_value = [
            MagicMock(page_content="Mocked context for testing")
        ]

        request = "Disassemble a laptop with 4 screws on the back panel"
        is_standard, validation_details, disassembly_plan = self.agent.handle_validate_request(request)

        self.assertTrue(is_standard)
        self.assertIn("standard procedure", validation_details.lower())
        self.assertIsNotNone(disassembly_plan)

    # Add more test methods here

if __name__ == '__main__':
    unittest.main()