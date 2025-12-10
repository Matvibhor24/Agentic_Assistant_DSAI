# app/models.py
from pydantic import BaseModel
from typing import List, Optional, Literal

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    # Not used directly in the endpoint now, but kept if you want a JSON version later
    text: Optional[str] = None
    thread_id: Optional[str] = None

class ExtractionResult(BaseModel):
    text: str
    source_type: Literal["text", "image", "pdf", "audio", "youtube", "unknown"]
    ocr_confidence: Optional[float] = None
    duration_seconds: Optional[float] = None

class Plan(BaseModel):
    task: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    reasoning: Optional[str] = None

class ChatResponse(BaseModel):
    extracted_text: str
    plan: Plan
    result: Optional[str] = None
    logs: List[str] = []
