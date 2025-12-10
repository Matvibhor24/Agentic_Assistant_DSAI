from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.state import AgentState, Task
from app.utils.llm import llm_json, chat_llm
from app.extractors.image_ocr import extract_image_text_from_bytes
from app.extractors.pdf_extractor import extract_pdf_text_from_bytes
from app.extractors.audio_transcriber import transcribe_audio_bytes
from app.tasks.summariser import summarize
from app.tasks.sentiment import analyze_sentiment
from app.tasks.code_explainer import explain_code
from app.tasks.qa import answer_question


def _get_last_user_content(messages) -> str:
    """
    LangGraph stores messages as LangChain Message objects (e.g. HumanMessage),
    not just dicts. Handle both shapes safely.
    """
    for m in reversed(messages or []):
        # Dict shape
        if isinstance(m, dict):
            if m.get("role") == "user":
                return m.get("content", "") or ""
        else:
            # LangChain message objects
            msg_type = getattr(m, "type", None)
            if msg_type == "human":
                return getattr(m, "content", "") or ""
    return ""


def start_node(state: AgentState) -> AgentState:
    """Start of each turn: just log that we started."""
    logs = state.get("logs", [])
    logs.append("Start node: starting new turn.")
    state["logs"] = logs
    return state


async def extract_node(state: AgentState) -> AgentState:
    """
    Decide how to extract content:
    - If file_bytes present: detect type and extract (image/pdf/audio).
    - Else: use last user message as extracted_text.
    """
    logs = state.get("logs", [])
    logs.append("Extract node: deciding how to extract content.")
    state["logs"] = logs

    file_bytes = state.get("file_bytes")
    file_name = (state.get("file_name") or "").lower()
    file_type = (state.get("file_content_type") or "").lower()

    # Case 1: File attached -> do extraction
    if file_bytes:
        if "image" in file_type or file_name.endswith((".png", ".jpg", ".jpeg")):
            text, _ = extract_image_text_from_bytes(file_bytes)
            state["extracted_text"] = text
            logs.append(f"Extract node: extracted text from image ({len(text)} chars).")

        elif "pdf" in file_type or file_name.endswith(".pdf"):
            text, _ = extract_pdf_text_from_bytes(file_bytes)
            state["extracted_text"] = text
            logs.append(f"Extract node: extracted text from PDF ({len(text)} chars).")

        elif "audio" in file_type or file_name.endswith((".mp3", ".wav", ".m4a")):
            text, dur = transcribe_audio_bytes(file_bytes, file_name or "audio")
            state["extracted_text"] = text
            logs.append(
                f"Extract node: transcribed audio ({len(text)} chars, duration ~{dur:.1f}s)."
            )
        else:
            logs.append("Extract node: unknown file type, fallback to text-only.")

        state["file_bytes"] = None

    if not state.get("extracted_text"):
        messages = state.get("messages", [])
        last_user = _get_last_user_content(messages)
        state["extracted_text"] = last_user
        logs.append("Extract node: using last user message as extracted_text.")

    state["logs"] = logs
    return state


async def planner_node(state: AgentState) -> AgentState:
    """
    Look at the latest user message and decide:
    - which task to run (summary, sentiment, etc.)
    - or whether we need clarification
    """
    logs = state.get("logs", [])
    logs.append("Planner node: inferring user intent.")
    state["logs"] = logs

    messages = state.get("messages", [])
    last_user = _get_last_user_content(messages)

    extracted = state.get("extracted_text", "")

    prompt = f"""
You are the PLANNER for an AI assistant.

Context:
- The assistant receives:
  - a user message (which may be empty if the user only uploaded a file), and
  - extracted text (from text, image OCR, PDF parsing, or audio transcription).
- Your job is to decide which ONE high-level task the assistant should perform,
  OR decide that the assistant must ask a clarification question before acting.

You DO NOT perform the task yourself. You ONLY choose the task and, if needed, a follow-up question.

--------------------
TASK OPTIONS
--------------------

You must choose exactly ONE of these task labels:

1) "summary"
   Use when the user clearly asks for a summary, TL;DR, brief version, or condensed explanation
   of the overall content.

2) "sentiment"
   Use when the user clearly asks about sentiment, mood, tone, positivity/negativity,
   or emotional attitude of the text.

3) "code_explanation"
   Use when the user clearly asks about code: what it does, how it works, bugs, or complexity.

4) "qa"
   Use when the user asks a specific question that should be answered from the content
   or involves explaining a concept mentioned in the content
   (e.g. "what is X?", "what do you mean by Y?", "what are the action items?").

5) "conversation"
   Use when the user is having a general chat or asking open-ended questions
   that are not strictly about analyzing the file content as a document
   (greetings, small talk, generic advice, etc.).

6) "transcript_only"
   Use when the user clearly requests only the raw transcription or extracted text,
   with no extra analysis (e.g. "just transcribe this", "give me the text only").

7) "none"
   Use only when you cannot safely choose any of the above even after thinking carefully.

--------------------
CLARIFICATION RULE
--------------------

You must also decide whether clarification is required.

1. If the user’s goal is clear and maps to exactly ONE task above:
   - Set "needs_clarification": false.
   - Choose that task.
   - Do NOT ask for a clarification.

2. If the user’s request is vague or incomplete AND *could reasonably correspond to multiple tasks*
   (for example, they only say "check this" or just upload a file with no instruction),
   THEN:
   - Set "needs_clarification": true.
   - You may set "task" to "none" OR to the best tentative task,
     but you MUST ask a clarification question before the assistant acts.

3. IMPORTANT DISTINCTION:
   - If the user asks a clear, direct question like:
     "what does this mean?", "what is globalization?", "what are the key ideas here?",
     this counts as a *clear QA-style request*, NOT as ambiguous.
     In such cases you should choose "qa" (or "conversation" if it is pure general chat)
     and set "needs_clarification": false.

4. Never ask for clarification just because a file is present.
   Only ask when the *intent* is genuinely ambiguous and you cannot tell
   whether they want summary, sentiment, code explanation, QA, etc.

5. Clarification behavior (when needed):
   - "clarification_question" must be short (1–2 sentences),
   - It should explicitly ask the user what they want you to do with the content
     (e.g. "Would you like a summary, a sentiment analysis, or answers to specific questions?").

--------------------
INPUT
--------------------

User message:
---
{last_user}
---

Extracted text (from their file or input):
---
{extracted[:4000]}
---

--------------------
OUTPUT FORMAT
--------------------

Return a SINGLE JSON object with the following shape:

{{
  "task": "summary" | "sentiment" | "code_explanation" | "qa" | "conversation" | "transcript_only" | "none",
  "needs_clarification": true or false,
  "clarification_question": "short question if needs_clarification is true, otherwise empty string",
  "reasoning": "a brief one-sentence explanation of why you chose this task or why you need clarification"
}}

Strict requirements:
- Do NOT include any text before or after the JSON.
- Do NOT include any extra keys.
- If needs_clarification is true, clarification_question must be non-empty.
- If needs_clarification is false, clarification_question must be an empty string.
"""


    result = await llm_json(prompt)

    task: Task = result.get("task", "none")
    needs_clar = bool(result.get("needs_clarification", False))
    question = result.get("clarification_question") or None
    reasoning = result.get("reasoning", "")

    state["task"] = task
    state["needs_clarification"] = needs_clar
    state["clarification_question"] = question

    logs.append(f"Planner chose task '{task}' (needs_clarification={needs_clar}).")
    logs.append(f"Planner reasoning: {reasoning}")
    state["logs"] = logs
    return state


def clarification_node(state: AgentState) -> AgentState:
    """
    If planner says we need clarification, we don't execute any tool.
    We simply add a follow-up question to messages.
    """
    logs = state.get("logs", [])
    logs.append("Clarification node: asking follow-up instead of executing.")
    state["logs"] = logs

    q = (
        state.get("clarification_question")
        or "What would you like me to do with this content?"
    )
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": q})
    state["messages"] = messages
    state["final_result"] = q
    return state


async def summary_node(state: AgentState) -> AgentState:
    text = state.get("extracted_text", "")
    out = await summarize(text)
    state["final_result"] = out
    logs = state.get("logs", [])
    logs.append("Summary node: generated multi-format summary.")
    state["logs"] = logs
    return state


async def sentiment_node(state: AgentState) -> AgentState:
    text = state.get("extracted_text", "")
    out = await analyze_sentiment(text)
    state["final_result"] = out
    logs = state.get("logs", [])
    logs.append("Sentiment node: computed sentiment.")
    state["logs"] = logs
    return state


async def code_explainer_node(state: AgentState) -> AgentState:
    text = state.get("extracted_text", "")
    out = await explain_code(text)
    state["final_result"] = out
    logs = state.get("logs", [])
    logs.append("Code explainer node: explained code and complexity.")
    state["logs"] = logs
    return state


async def qa_node(state: AgentState) -> AgentState:
    text = state.get("extracted_text", "")
    messages = state.get("messages", [])
    last_user = _get_last_user_content(messages)
    out = await answer_question(text, last_user)
    state["final_result"] = out
    logs = state.get("logs", [])
    logs.append("QA node: answered question based on context.")
    state["logs"] = logs
    return state


async def conversation_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    out = await chat_llm(messages)
    state["final_result"] = out
    logs = state.get("logs", [])
    logs.append("Conversation node: responded conversationally.")
    state["logs"] = logs
    return state


def transcript_only_node(state: AgentState) -> AgentState:
    text = state.get("extracted_text", "")
    state["final_result"] = text
    logs = state.get("logs", [])
    logs.append("Transcript-only node: returning transcript as-is.")
    state["logs"] = logs
    return state


def finalize_node(state: AgentState) -> AgentState:
    logs = state.get("logs", [])
    logs.append("Finalize node: done.")
    state["logs"] = logs
    return state


# ---------- Routing after planner ----------


def route_after_planner(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "clarification"
    task = state.get("task", "none")
    if task == "summary":
        return "summary"
    if task == "sentiment":
        return "sentiment"
    if task == "code_explanation":
        return "code_explainer"
    if task == "qa":
        return "qa"
    if task == "conversation":
        return "conversation"
    if task == "transcript_only":
        return "transcript_only"
    return "conversation"


# ---------- Build graph with checkpointing ----------

workflow = StateGraph(AgentState)

workflow.add_node("start", start_node)
workflow.add_node("extract", extract_node)
workflow.add_node("planner", planner_node)
workflow.add_node("clarification", clarification_node)

workflow.add_node("summary", summary_node)
workflow.add_node("sentiment", sentiment_node)
workflow.add_node("code_explainer", code_explainer_node)
workflow.add_node("qa", qa_node)
workflow.add_node("conversation", conversation_node)
workflow.add_node("transcript_only", transcript_only_node)

workflow.add_node("finalize", finalize_node)

workflow.add_edge(START, "start")
workflow.add_edge("start", "extract")
workflow.add_edge("extract", "planner")

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "clarification": "clarification",
        "summary": "summary",
        "sentiment": "sentiment",
        "code_explainer": "code_explainer",
        "qa": "qa",
        "conversation": "conversation",
        "transcript_only": "transcript_only",
    },
)

for node in [
    "summary",
    "sentiment",
    "code_explainer",
    "qa",
    "conversation",
    "transcript_only",
]:
    workflow.add_edge(node, "finalize")

workflow.add_edge("clarification", END)
workflow.add_edge("finalize", END)

checkpointer = MemorySaver()
agent_app = workflow.compile(checkpointer=checkpointer)
