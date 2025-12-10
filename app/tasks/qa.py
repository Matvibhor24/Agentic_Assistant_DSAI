from app.utils.llm import chat_llm

async def answer_question(context: str, question: str) -> str:
    prompt = f"""
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer based only on the context. If you don't know, say you don't know.
"""
    return await chat_llm([{"role": "user", "content": prompt}])
