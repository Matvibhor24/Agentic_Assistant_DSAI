# app/tasks/summarizer.py
from app.utils.llm import chat_llm

async def summarize(text: str) -> str:
    prompt = f"""
You are a concise summarizer.

Summarize the following text in three formats:

1) One-line summary
2) Three bullet points
3) Five-sentence detailed summary

Text:
{text}
"""
    content = await chat_llm([{"role": "user", "content": prompt}])
    return content
