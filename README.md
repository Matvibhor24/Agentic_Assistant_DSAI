# Agentic Assistant (DSAI)

A simple AI helper that can read text, images, PDFs, and audio, figure out what you want, and return a plain-text reply. If it is unsure, it asks you a follow-up question instead of guessing.

## What it does

- Upload a file (image, PDF, or audio) or just type a message.
- The app extracts the content, plans a task, and replies in plain text.
- No code is needed—everything runs through a Streamlit UI.

## Quick start

1. Install Python 3.10+ and Git.
2. Clone the repo:  
   `git clone https://github.com/Matvibhor24/Agentic_Assistant_DSAI.git`  
   `cd Agentic_Assistant_DSAI`
3. Create a virtual environment and install deps:  
   `python -m venv venv`  
   `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)  
   `pip install -r requirements.txt`
4. Start the backend API:  
   `uvicorn app.main:app --reload`
5. In a second terminal, start the Streamlit UI:  
   `streamlit run streamlit_app.py`
6. Open the Streamlit link shown in the terminal (usually `http://localhost:8501`), upload a file or type a prompt, and read the reply.

## Notes

- Keep the backend running while you use the UI.
- Responses are text-only (no images or rich formatting).
