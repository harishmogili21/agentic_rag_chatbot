import streamlit as st
import os
from dotenv import load_dotenv
import json

from utils.mcp import MCPMessage
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.response_agent import LLMResponseAgent

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_MIN_LOG_LEVEL"] = "3"

# --- CONFIGURATION ---
load_dotenv()
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- COORDINATOR ---
class Coordinator:
    """
    The central coordinator that routes messages between agents.
    This implements the in-memory pub/sub mechanism for MCP.
    """
    def __init__(self):
        self.agents = {}
        self.ui_callback = None

    def register_agent(self, agent):
        self.agents[agent.name] = agent

    def set_ui_callback(self, callback):
        self.ui_callback = callback

    def send(self, message: MCPMessage):
        """Routes a message to the appropriate agent or the UI."""
        
        try:
            print('#'*60)
            msg_dict = message.model_dump() if hasattr(message, 'model_dump') else message.__dict__
            print("\n[Coordinator] MCP message (pretty print):\n" + json.dumps(msg_dict, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"[Coordinator] MCP message (raw): {message}\n[Pretty print error: {e}]")
        print(f"Coordinator routing message from {message.sender} to {message.receiver} (Type: {message.type})")
        print('#'*60)
        if message.receiver in self.agents:
            self.agents[message.receiver].process_message(message)
        elif message.receiver == "Coordinator":
            # Message is for the coordinator itself to process and re-route
            if message.type == "EMBED_REQUEST":
                self.send(MCPMessage(sender="Coordinator", receiver="RetrievalAgent", type="EMBED_REQUEST", payload=message.payload))
            elif message.type == "RETRIEVAL_RESPONSE":
                # Use 'retrieved_context' from RetrievalAgent's payload
                self.send(MCPMessage(
                    sender="Coordinator",
                    receiver="LLMResponseAgent",
                    type="GENERATE_REQUEST",
                    payload={
                        "query": message.payload.get("query"),
                        "context_chunks": message.payload.get("retrieved_context", []),
                        "trace_id": message.payload.get("trace_id")
                    }
                ))
            elif message.type == "INGEST_COMPLETE":
                 if self.ui_callback:
                    self.ui_callback("ingest_complete", {})
            elif message.type == "GENERATE_RESPONSE":
                if self.ui_callback:
                    self.ui_callback("final_answer", message.payload)
        else:
            print(f"Warning: No agent or handler registered for receiver '{message.receiver}'")

# --- STREAMLIT UI ---

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
st.title("🤖 Agentic RAG Chatbot with MCP")

# Initialize session state
if "coordinator" not in st.session_state:
    st.session_state.coordinator = Coordinator()

    # Define the UI callback function
    def ui_callback_handler(event_type, payload):
        if event_type == "ingest_complete":
            st.session_state.processing_files = False
            st.session_state.files_processed = True
            st.rerun()
        elif event_type == "final_answer":
            st.session_state.messages.append({"role": "assistant", "content": payload["answer"], "sources": payload["sources"]})
            st.session_state.processing_query = False
            st.rerun()

    st.session_state.coordinator.set_ui_callback(ui_callback_handler)

    # Register agents
    st.session_state.coordinator.register_agent(IngestionAgent(st.session_state.coordinator.send))
    # Patch: Check if FAISS DB exists and set files_processed accordingly
    retrieval_agent = RetrievalAgent(st.session_state.coordinator.send)
    st.session_state.coordinator.register_agent(retrieval_agent)
    if getattr(retrieval_agent, 'index', None) is not None and getattr(retrieval_agent, 'chunks_with_metadata', []):
        st.session_state.files_processed = True
    st.session_state.coordinator.register_agent(LLMResponseAgent(st.session_state.coordinator.send))
    print("Coordinator and Agents initialized.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False
if "processing_files" not in st.session_state:
    st.session_state.processing_files = False
if "processing_query" not in st.session_state:
    st.session_state.processing_query = False

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, PPTX, CSV, or TXT files",
        type=["pdf", "docx", "pptx", "csv", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.processing_files:
        st.session_state.processing_files = True
        file_paths = []
        skipped_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            if os.path.exists(file_path):
                skipped_files.append(uploaded_file.name)
                continue
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        if skipped_files:
            st.sidebar.warning(f"Skipped duplicate files: {', '.join(skipped_files)}")
        if file_paths:
            st.info("Processing uploaded files... Please wait.")
            # Start ingestion process
            ingest_message = MCPMessage(
                sender="UI",
                receiver="IngestionAgent",
                type="INGEST_REQUEST",
                payload={"file_paths": file_paths}
            )
            st.session_state.coordinator.send(ingest_message)
        else:
            st.session_state.processing_files = False
            if skipped_files:
                st.sidebar.info("All selected files were already uploaded.")
                
if st.session_state.files_processed:
    st.sidebar.success("✅ Files processed and ready!")


# Main chat interface
if not st.session_state.files_processed and not st.session_state.processing_files:
    st.info("Please upload documents in the sidebar to begin.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"Source {i+1}:\n\n" + source.strip())

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.files_processed:
        st.warning("Please upload and process documents before asking a question.")
    else:
        st.session_state.processing_query = True
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Send retrieval request
        retrieval_message = MCPMessage(
            sender="UI",
            receiver="RetrievalAgent",
            type="RETRIEVAL_REQUEST",
            payload={"query": prompt}
        )
        st.session_state.coordinator.send(retrieval_message)

if st.session_state.processing_query:
    st.spinner("Thinking...")