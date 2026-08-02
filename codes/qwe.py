from groq import Groq
import base64
import os

api_key=os.getenv("GROQ_API_KEY")
client = Groq(api_key)

IMAGE_PATH = r"dataset\receipt1.jpg" #change this to the respective receipt num

with open(IMAGE_PATH, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")


prompt = """
Extract the receipt details.

Return ONLY valid JSON:

{
    "receipt_number": "",
    "vendor": "",
    "date": "",
    "amount": "",
    "currency": "",
    "gst": "",
    "invoice_no": ""
}
"""


completion = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
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
    ],
    temperature=0.2,
    max_completion_tokens=1024
)


import os

OUTPUT_PATH = r"outputs3\receipt1.json" #change this to the respected receipt number

result = completion.choices[0].message.content

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(result)

print("Saved receipt1.json")