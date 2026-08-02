import os

import cohere
import base64
import json

co = cohere.ClientV2(
    api_key=os.getenv("COHERE_API_KEY")
)
IMAGE_PATH = r"dataset\receipt1.jpg" #change this to the respective receipt num

from PIL import Image
import io
import base64

img = Image.open(IMAGE_PATH)

#to resize if large
img.thumbnail((1600, 1600))

buffer = io.BytesIO()
img.save(buffer, format="JPEG")

image_base64 = base64.b64encode(
    buffer.getvalue()
).decode("utf-8")

prompt = """
Extract information from this receipt.

Return ONLY a valid JSON object.
Do not use markdown.
Do not use code fences.
Do not add explanations or extra text.

Use exactly this schema:

{
    "vendor": "",
    "date": "",
    "amount": "",
    "currency": "",
    "gst": "",
    "invoice_no": ""
}

Rules:
- date must be formatted as YYYY-MM-DD.
- amount must contain only the numeric value (example: 360, not "Rs.360").
- currency should contain the currency symbol/code if visible (example: Rs., INR, ₹).
- If a field is missing or unclear, return an empty string.
- Do not guess or invent values.
- Return only the JSON object.
"""

response = co.chat(
    model="command-a-vision-07-2025",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
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
)

result = response.message.content[0].text

prediction = json.loads(result)

with open("outputs2/receipt1.json", "w", encoding="utf-8") as f: #change the oututs/receiptnumber.json to the respected receipt number
    json.dump(prediction, f, indent=4)
