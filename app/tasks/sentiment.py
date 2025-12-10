from app.utils.llm import chat_llm

async def analyze_sentiment(text: str) -> str:
    prompt = f"""
Analyze the sentiment of the following text.

Return:
- Label: Positive / Negative / Neutral
- Confidence: 0-100
- One-line justification.

Text:
{text}
"""
    return await chat_llm([{"role": "user", "content": prompt}])
