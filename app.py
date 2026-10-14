"""
Frontend: Streamlit UI that talks to the FastAPI backend.

Run (from the frontend/ folder):
    streamlit run app.py
"""

import os
import requests
import streamlit as st

DEFAULT_BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_MODEL = "openai/gpt-oss-120b"


def api_error(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


def main():
    st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎬", layout="wide")
    st.title("🎬 YouTube RAG Chatbot")
    st.caption("Streamlit frontend → FastAPI backend → LangChain + FAISS + Groq")

    with st.sidebar:
        st.header("Configuration")
        backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND).rstrip("/")
        api_key = st.text_input(
            "Groq API key (optional if set on server)",
            type="password",
            help="Create a key at https://console.groq.com/keys",
        )
        model_name = st.text_input("Groq model name", value=DEFAULT_MODEL)

        if st.button("Check backend"):
            try:
                r = requests.get(f"{backend_url}/health", timeout=5)
                st.success(f"Backend OK: {r.json()}")
            except Exception as e:
                st.error(f"Cannot reach backend: {e}")

    for key, default in {
        "session_id": None,
        "video_id": None,
        "chunk_count": 0,
        "chat_history": [],
    }.items():
        st.session_state.setdefault(key, default)

    # ---- Index a video ----
    with st.form("load_video_form"):
        youtube_url = st.text_input(
            "YouTube video URL", placeholder="https://www.youtube.com/watch?v=..."
        )
        load_clicked = st.form_submit_button("Load and index video", type="primary")

    if load_clicked:
        if not youtube_url.strip():
            st.error("Please enter a YouTube URL.")
        else:
            with st.spinner("Fetching transcript and building index..."):
                try:
                    resp = requests.post(
                        f"{backend_url}/index",
                        json={
                            "url": youtube_url,
                            "api_key": api_key or None,
                            "model_name": model_name or None,
                        },
                        timeout=300,
                    )
                    if resp.ok:
                        data = resp.json()
                        # Free the previous session on the server
                        if st.session_state.session_id:
                            try:
                                requests.delete(
                                    f"{backend_url}/session/{st.session_state.session_id}",
                                    timeout=5,
                                )
                            except Exception:
                                pass
                        st.session_state.session_id = data["session_id"]
                        st.session_state.video_id = data["video_id"]
                        st.session_state.chunk_count = data["chunk_count"]
                        st.session_state.chat_history = []
                        st.success(
                            f"Indexed {data['chunk_count']} chunks "
                            f"(transcript language: {data['language']})."
                        )
                    else:
                        st.error(f"Indexing failed: {api_error(resp)}")
                except requests.RequestException as e:
                    st.error(f"Cannot reach backend at {backend_url}: {e}")

    # ---- Chat ----
    if st.session_state.session_id:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        st.info(f"Ready to chat. Indexed {st.session_state.chunk_count} transcript chunks.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("View retrieved transcript excerpts"):
                        for i, s in enumerate(msg["sources"], start=1):
                            st.markdown(f"**Excerpt {i}**")
                            st.write(s)

        question = st.chat_input("Ask a question about this video...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(
                            f"{backend_url}/chat",
                            json={
                                "session_id": st.session_state.session_id,
                                "question": question,
                            },
                            timeout=120,
                        )
                        if resp.ok:
                            data = resp.json()
                            st.markdown(data["answer"])
                            with st.expander("View retrieved transcript excerpts"):
                                for i, s in enumerate(data["sources"], start=1):
                                    st.markdown(f"**Excerpt {i}**")
                                    st.write(s)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": data["answer"],
                                "sources": data["sources"],
                            })
                        else:
                            st.error(f"Error: {api_error(resp)}")
                    except requests.RequestException as e:
                        st.error(f"Cannot reach backend: {e}")


if __name__ == "__main__":
    main()
