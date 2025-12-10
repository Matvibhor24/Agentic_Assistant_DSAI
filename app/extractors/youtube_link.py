from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def fetch_youtube_transcript(url: str) -> str:
    vid = extract_video_id(url)
    if not vid:
        return "Could not detect YouTube video id."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        text = " ".join([t["text"] for t in transcript])
        return text
    except Exception:
        return "Transcript not available for this YouTube video."
