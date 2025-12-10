import base64
from typing import Tuple

from app.utils.llm import client
from app.utils.config import OPENAI_MODEL


def extract_image_text_from_bytes(data: bytes) -> Tuple[str, float]:
    """
    Use OpenAI vision (gpt-4o*) to read text from an image.
    Returns text and a dummy confidence (1.0 on success).
    """
    b64 = base64.b64encode(data).decode("utf-8")
    image_url = f"data:image/png;base64,{b64}"

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all visible text from this image. Return plain text only.",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=0,
    )

    text = resp.choices[0].message.content or ""
    return text.strip(), 1.0
