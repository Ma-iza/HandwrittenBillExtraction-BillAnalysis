import base64
import requests
import time
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

IMAGE_PATH = r"dataset\receipt1.jpg" #change this to the respective receipt num

OUTPUT_PATH = r"outputs\receipt1.json" #change this to the respected receipt number

MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"


PROMPT = """
Look at this receipt image and extract the following fields.

Return ONLY valid JSON.

{
    "vendor": "",
    "date": "",
    "amount": "",
    "currency": "",
    "gst": "",
    "invoice_no": ""
}

Rules:
- Normalize date to YYYY-MM-DD.
- Amount should be numeric.
"""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


image_base64 = encode_image(IMAGE_PATH)


response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    },
    timeout=180
)

for attempt in range(3):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            },
            timeout=180
        )

        if response.status_code == 200:
            data = response.json()

            if "choices" in data:
                result = data["choices"][0]["message"]["content"]

                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    f.write(result)

                print("Saved receipt4.json")
                break

        print("Failed:", response.text)

    except requests.exceptions.Timeout:
        print(f"Attempt {attempt+1} timed out")

    time.sleep(5)

if response.status_code == 200:
    data = response.json()

    if "choices" in data:
        result = data["choices"][0]["message"]["content"]

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(result)

        print("Saved receipt3.json")

    else:
        print("Unexpected response:")
        print(data)

else:
    print("API Error:")
    print(response.text)
