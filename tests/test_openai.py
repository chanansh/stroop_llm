from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()

def test_openai_api_key_works():
    assert os.getenv("OPENAI_API_KEY") is not None
    # test that the api key is valid
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    assert client is not None
    # test client with a sample request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}]
    )
    assert response is not None
