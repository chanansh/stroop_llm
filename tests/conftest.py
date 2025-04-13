import os
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set test-specific environment variables
    os.environ["OPENAI_API_KEY"] = "test_key"
    
    # Reset any test-specific state
    yield
    
    # Cleanup after each test
    pass

@pytest.fixture
def mock_openai_client(mocker):
    """Create a mock OpenAI client for testing."""
    mock_client = mocker.Mock()
    mock_client.get_response.return_value = ("b", {
        "api_time": 0.5,
        "total_time": 0.6,
        "retry_count": 0
    })
    mock_client.get_feedback_response.return_value = "Thank you for the feedback!"
    return mock_client 