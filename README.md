# Agentic_Assistant_DSAI
The assistant can accept text, images, PDFs, and audio, extract the content, understand the user’s goal, plan the correct task, and produce a text-only final answer. If the goal is not clear, it does not guess — it asks a follow-up question before performing any task.
Agentic Multimodal Assistant

This is a simple AI agent that can understand and process different types of inputs:

Text

Image (OCR)

PDF (text-based + scanned)

Audio (speech-to-text)

The agent extracts content itself, understands the user’s goal, and performs the correct task automatically.
If the request is not clear, it asks a follow-up question instead of guessing.

All final outputs are text-only.

Features

Text chat (general conversation)

Summarization (1-line + bullets + 5-sentences)

Sentiment analysis (label + confidence + justification)

Code explanation (OCR → explain code + find bugs + time complexity)

QA from the extracted content

Transcript-only output when asked

Clarification questions when the intent is unclear

Execution logs for explainability

Tech Stack

FastAPI – backend API

Streamlit – simple chat UI

LangGraph – agent planning & state handling

OCR / PDF Parsing / Speech-to-Text – automatic extraction

How to Run
# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn app.main:app --reload

# In a new terminal, run UI
streamlit run streamlit_app.py


Open → http://localhost:8501

Quick Test Instructions

1️⃣ Upload a PDF only → agent should ask what to do
2️⃣ Tell it to summarize → summary appears
3️⃣ Ask questions about the PDF → QA working
4️⃣ Upload a code image → code explanation + bug detection
5️⃣ Upload audio → transcription + answer concept question
6️⃣ Say “thanks” → normal chat response
7️⃣ Say “check this” → agent should ask clarification
