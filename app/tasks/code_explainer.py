from app.utils.llm import chat_llm

async def explain_code(code: str) -> str:
    prompt = f"""
You are a senior software engineer.

Given the following code, do the following:

1) Briefly explain what it does.
2) Point out any obvious bugs or risky assumptions.
3) Give time and space complexity in Big-O.

Code:
```code
{code}
```
"""
    return await chat_llm([{"role": "user", "content": prompt}])