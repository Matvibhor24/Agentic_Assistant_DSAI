from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from app.models import ChatResponse, Plan
from app.state import AgentState
from app.graph import agent_app

app = FastAPI(title="DataSmith Agent (Extraction by Agent)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    text: Optional[str] = Form(None),
    thread_id: str = Form("default-thread"),
    file: Optional[UploadFile] = File(None),
):
    """
    Main endpoint:
    - If file present: pass its bytes + metadata into state.
    - Agent (graph) decides how to extract (image/pdf/audio or pure text).
    - LangGraph checkpointing keeps conversation memory per thread_id.
    """
    logs = []

    messages = []
    if text:
        messages.append({"role": "user", "content": text})
    else:
        messages.append({"role": "user", "content": ""})

    file_bytes: Optional[bytes] = None
    file_name: Optional[str] = None
    file_content_type: Optional[str] = None

    if file:
        file_bytes = await file.read()
        file_name = file.filename
        file_content_type = file.content_type or ""
        logs.append(f"FastAPI: received file {file_name} ({file_content_type}).")

    state: AgentState = {
        "messages": messages,
        "logs": logs,
        "file_bytes": file_bytes,
        "file_name": file_name,
        "file_content_type": file_content_type,
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = await agent_app.ainvoke(state, config=config)

    final_extracted = final_state.get("extracted_text", "")
    final_logs = final_state.get("logs", [])
    task = final_state.get("task", "none")
    needs_clar = final_state.get("needs_clarification", False)
    clar_q = final_state.get("clarification_question")
    result = final_state.get("final_result")

    plan = Plan(
        task=task,
        needs_clarification=needs_clar,
        clarification_question=clar_q,
        reasoning=None,
    )

    return ChatResponse(
        extracted_text=final_extracted,
        plan=plan,
        result=result,
        logs=final_logs,
    )
