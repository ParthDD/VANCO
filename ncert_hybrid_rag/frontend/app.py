import streamlit as st
import requests

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="NCERT Physics RAG", layout="wide")

st.title("NCERT Physics Hybrid RAG")
st.caption("Real-time question answering grounded in the NCERT PDF")

backend_url = "http://127.0.0.1:8000"

try:
    response = requests.get(f"{backend_url}/health", timeout=5)
    if response.status_code == 200:
        st.success("Backend connected successfully")
    else:
        st.error("Backend is running but health check failed")
except Exception:
    st.error("Backend not connected")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question from NCERT Physics...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        api_response = requests.get(
            f"{backend_url}/query/ask",
            params={"question": prompt, "top_k": 5},
            timeout=60
        )

        if api_response.status_code == 200:
            data = api_response.json()

            logger.info("Question: %s", prompt)
            logger.info("Raw API response: %s", data)
            logger.info("Answer: %s", data.get("answer"))
            logger.info("Citations: %s", data.get("citations", []))

            answer = data.get("answer", "No answer returned.")

            citations = data.get("citations", [])
            if citations:
                citation_text = "\n\n**Citations:**\n" + "\n".join(
                    [
                        f"- Source {c.get('source_number', '?')} | Page {c.get('page_number', '?')} | Chunk {c.get('chunk_id', 'N/A')}"
                        for c in citations
                    ]
                )
            else:
                citation_text = "\n\n**Citations:** None returned."

            reply = answer + citation_text
        else:
            reply = f"Query failed with status code {api_response.status_code}"

    except Exception as e:
        reply = f"Error connecting to backend query API: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)