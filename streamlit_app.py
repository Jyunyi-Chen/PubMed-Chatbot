import json
import time
import uuid
import base64
import streamlit as st

from pathlib import Path
from chatbot import Chatbot

from streamlit_functions import load_cache_resource
from streamlit_functions import warmup_ollama_models

st.set_page_config(page_title="PubMed Chatbot", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] h1 { font-size: 20px !important; }
    .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, 
    .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a { display: none; }
    [data-testid="stChatInput"] textarea::placeholder { font-weight: bold; }
    .stMarkdown h1 { font-size: 1.5rem !important; margin-bottom: 0.5rem !important; }
    .stMarkdown h2 { font-size: 1.3rem !important; margin-bottom: 0.5rem !important; margin-top: 1.5rem !important; }
    .stMarkdown h3 { font-size: 1.15rem !important; margin-bottom: 0.5rem !important; margin-top: 1.5rem !important; }
    .stMarkdown p { font-size: 1rem !important; line-height: 1.6 !important; margin-bottom: 0.8rem !important; }
    .stMarkdown ul, .stMarkdown ol { margin-bottom: 1.2rem !important; }
    .stMarkdown li { margin-bottom: 0.3rem !important; line-height: 1.6 !important; }
    </style>
    """,
    unsafe_allow_html=True
)


def make_graph_link_html(html_content: str) -> str:
    """Generate an HTML snippet that opens the knowledge graph in a new tab using a Blob URL."""
    b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    return f"""
    <a href="#" id="graphLink" 
       style="font-family:sans-serif;font-size:18px;font-weight:bold;color:#0066cc;
              text-decoration:underline;cursor:pointer;">
       Open Knowledge Graph
    </a>
    <script>
        var b64 = "{b64}";
        var raw = atob(b64);
        var blob = new Blob([raw], {{type: "text/html"}});
        var url = URL.createObjectURL(blob);
        var link = document.getElementById("graphLink");
        link.href = url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
    </script>
    """


with st.spinner("Loading UMLS database & Knowledge Graphs ... This might take a minute."):

    ner_pipeline, topic_to_pmid_to_node_weights, topic_to_graphs = load_cache_resource()

    warmup_ollama_models()

with st.sidebar:
    
    chat_topic = st.selectbox("**Chat Topic**", ["MRT", "MSC", "TRIP", "TWHM"])

    n_max_refs = st.selectbox("**Max References**", [10, 20, 30, 40, 50])

    rag_options = ["Text", "Graph", "Hybrid"]

    prev_rag_scheme = st.session_state.get("rag_scheme", None)

    if prev_rag_scheme in rag_options: default_index = rag_options.index(prev_rag_scheme)
    else: default_index = 0

    rag_scheme = st.selectbox("**RAG Scheme**", rag_options, index=default_index)
    st.session_state["rag_scheme"] = rag_scheme

    chat_model = st.selectbox("**Chat Model**", ["GPT-5.2", "GPT-4.1", "GPT-OSS-20B"])
    prefer_lan = st.selectbox("**Preferred Response Language**", ["English", "Chinese"])

if ("chatbot" not in st.session_state) or (st.session_state.get("curr_chat_topic") != chat_topic):

    st.session_state.chatbot = Chatbot(chat_topic, ner_pipeline, topic_to_pmid_to_node_weights, topic_to_graphs)
    
    if "chat_history" not in st.session_state: st.session_state.chat_id, st.session_state.chat_history = str(uuid.uuid4()), []

    st.session_state.curr_chat_topic = chat_topic
    
for query, response, graph_b64 in st.session_state.chat_history:

    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        
        st.markdown(response)

        if graph_b64 != "N/A":

            html_content = base64.b64decode(graph_b64).decode("utf-8")
            link_html = make_graph_link_html(html_content)
            st.components.v1.html(link_html, height=40)

if query := st.chat_input("Enter your question ..."):

    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        start_time = time.time()

        with st.spinner("Generating response ..."):

            chat_log: dict[str, str] = st.session_state.chatbot.get_response(
                st.session_state.chat_id, query, rag_scheme,
                n_max_refs, chat_model, prefer_lan
            )

        response = f"*Response generation completed, {(time.time() - start_time):.2f} seconds taken.*\n\n{chat_log['response']}"

        if chat_log["chat_model"] != "N/A": response += f"\n\n*Answered by {chat_log['chat_model']}.*"

        message_placeholder.markdown(response)

        digraph_html_path = Path(chat_log["digraph_html_path"])

        try:

            html_content: str = digraph_html_path.read_text(encoding="utf-8")
            link_html = make_graph_link_html(html_content)
            st.components.v1.html(link_html, height=40)

            graph_b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
            st.session_state.chat_history.append((query, response, graph_b64))

        except:

            st.session_state.chat_history.append((query, response, "N/A"))