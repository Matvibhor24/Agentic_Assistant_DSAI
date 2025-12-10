import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
API_CHAT = f"{API_BASE}/api/chat"

st.set_page_config(page_title="Agentic Assistant", page_icon="💬", layout="centered")

# Force any pre/code blocks (e.g., model replies with ``` or OCR text) to wrap
# so the message is shown normally instead of inside a horizontal scrollbar.
st.markdown(
    """
    <style>
    div[data-testid="stMarkdownContainer"] pre,
    div[data-testid="stMarkdownContainer"] code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- SESSION STATE ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # for UI only: [{"role": "user"|"assistant", "content": str}]
    st.session_state.messages = []


# ---------- HEADER ----------
st.title("💬 Agentic Assistant")
st.caption(
    "Upload text, image, PDF, or audio · Agent extracts, understands, and responds."
)

if st.button("New chat"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()


# ---------- CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------- INPUT AREA ----------
uploaded_file = st.file_uploader(
    "Attach a file (optional)",
    type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "m4a"],
    label_visibility="collapsed",
)

user_input = st.chat_input("Type your message here (you can also attach a file above)")

# IMPORTANT CHANGE:
# We now trigger a send ONLY when user_input is not None
# (i.e., when the user hits Enter in the chat box).
if user_input is not None:
    # If user somehow hits enter with nothing and no file
    if not user_input.strip() and uploaded_file is None:
        st.warning("Please type a message or attach a file.")
        st.stop()

    # -------- USER MESSAGE IN UI --------
    if uploaded_file is not None and not user_input.strip():
        # (This case is unlikely because chat_input won't send empty, but kept for safety)
        user_text = f"📎 {uploaded_file.name}"
    elif uploaded_file is not None:
        user_text = f"{user_input}\n\n📎 {uploaded_file.name}"
    else:
        user_text = user_input

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # -------- CALL BACKEND --------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = {"thread_id": st.session_state.thread_id}
            if user_input.strip():
                data["text"] = user_input

            files = None
            if uploaded_file is not None:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }

            try:
                resp = requests.post(API_CHAT, data=data, files=files)
            except Exception as e:
                assistant_text = f"❌ Could not reach backend: `{e}`"
                st.markdown(assistant_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                st.stop()

            if resp.status_code != 200:
                assistant_text = f"❌ Backend error {resp.status_code}: {resp.text}"
                st.markdown(assistant_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                st.stop()

            data = resp.json()

            extracted = data.get("extracted_text") or ""
            result = data.get("result") or ""
            plan = data.get("plan") or {}
            clar_q = plan.get("clarification_question")

            parts = []

            if extracted:
                trimmed = (
                    extracted if len(extracted) <= 300 else extracted[:300] + "..."
                )
                parts.append("**Extracted text (snippet):**")
                parts.append(f"> {trimmed}")

            if result:
                parts.append("**Response:**")
                parts.append(result)
            elif clar_q:
                parts.append("**Clarification:**")
                parts.append(clar_q)
            else:
                parts.append("_No response generated._")

            assistant_text = "\n\n".join(parts)
            st.markdown(assistant_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )

    # -------- LOGS SECTION --------
    logs = data.get("logs", [])
    if logs:
        with st.expander("🔍 Execution Logs"):
            for log in logs:
                st.markdown(f"- {log}")
