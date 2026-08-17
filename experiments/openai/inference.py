import base64
import os
from openai import OpenAI

with open("token.txt") as data:
    openai_api_key = data.read().strip()

openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

response = client.chat.completions.create(
    model="OpenGVLab/InternVL3-78B",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    'image_url': {
                        'url':
                        'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg',
                    },
                },
            ],
        },
    ]
)
print("OpenGV-InternVL3 Vision Output:")
print(response.choices[0].message.content)
