from typing import Tuple
import io
from openai import OpenAI

from app.utils.config import OPENAI_API_KEY, WHISPER_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio_bytes(data: bytes, filename: str) -> Tuple[str, float]:
    """
    Transcribe audio using OpenAI Whisper without relying on ffmpeg/pydub.
    Duration is set to 0.0 (not critical for downstream use).
    """
    temp_file = io.BytesIO(data)
    temp_file.name = filename

    transcript = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=temp_file,
        response_format="text",
    )
    text = transcript.strip()
    return text, 0.0
