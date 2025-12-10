from typing import List, Optional, Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

Task = Literal[
    "summary",
    "sentiment",
    "code_explanation",
    "qa",
    "conversation",
    "transcript_only",
    "none",
]

class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

    extracted_text: str

    task: Task
    needs_clarification: bool
    clarification_question: Optional[str]

    final_result: str

    logs: List[str]

    file_bytes: Optional[bytes]
    file_name: Optional[str]
    file_content_type: Optional[str]
