import base64
import requests
import os
import dotenv

dotenv.load_dotenv()

API_URL = "https://router.huggingface.co/hf-inference/models/openai/clip-vit-base-patch32"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

def query(data):
    with open(data["image_path"], "rb") as f:
        img = f.read()
    payload={
        "parameters": data["parameters"],
        "inputs": base64.b64encode(img).decode("utf-8")
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "image_path": "stimuli_images/BLUE_0_0_255.png",
    "parameters": {"candidate_labels": ["blue", "red"]},
})

print(output)