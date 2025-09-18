# Agentic RAG Chatbot with Google Gemini

## 📝 Overview

This project is a solution to the "Agentic RAG Chatbot" coding challenge. It implements a sophisticated, agent-based Retrieval-Augmented Generation (RAG) system capable of answering questions based on user-uploaded documents in various formats (PDF, DOCX, PPTX, CSV, TXT).

The core of the architecture is a multi-agent system where agents communicate using a structured message-passing system called the Model Context Protocol (MCP). This version has been updated to use **Google's Gemini models** for response generation and **huggingface** for embeddings.

---

## ✨ Core Features

* **Multi-Format Document Support**: Upload and process PDF, DOCX, PPTX, CSV, and TXT/Markdown files.
* **Agentic Architecture**: A modular system with three distinct agents:
    * `IngestionAgent`: Parses and chunks documents.
    * `RetrievalAgent`: Creates vector embeddings and retrieves relevant context.
    * `LLMResponseAgent`: Generates answers based on the retrieved context.
* **Model Context Protocol (MCP)**: Agents communicate via a structured, in-memory messaging protocol for clear and traceable workflows.
* **Vector Search**: Uses `FAISS` for efficient in-memory similarity search.
* **Powered by Google Gemini**:
    * **Embeddings**: `all-MiniLM-L6-v2` for generating dense vector representations.
    * **LLM**: `gemini-2.0-flash` for fast and powerful response generation.
* **Interactive UI**: A user-friendly chat interface built with Streamlit that allows document uploads, multi-turn conversations, and displays source context for answers.

---

## 🏗️ Architecture & System Flow

The application follows a coordinator-agent pattern. The Streamlit UI acts as the entry point, passing user actions to a `Coordinator` which then routes messages between the specialized agents.

## 🛠️ Tech Stack

* **UI Framework**: Streamlit
* **Core Logic**: Python 3.9+
* **LLM & Embeddings**: Google Gemini (`gemini-2.0-flash`), Sentence Transformers(`all-MiniLM-L6-v2`)
* **Vector Store**: FAISS (CPU)
* **Document Parsing**: `pypdf`, `python-docx`, `python-pptx`, `pandas`
* **Text Processing**: LangChain (for text splitting)
* **Data Validation**: Pydantic (for MCP)

---

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd < repo_name >
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

This project uses the Google Gemini API.

1.  Create a `.env` file in the root directory of the project.
2.  Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
3.  Add your key to the `.env` file:

    ```env
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

---

## ▶️ How to Run the Application

Once you have completed the setup, you can run the Streamlit application with a single command:

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

**How to Use the App:**
1.  Upload one or more documents using the sidebar uploader.
2.  Wait for the "Files processed and ready!" confirmation message.
3.  Type your question in the chat input at the bottom of the page and press Enter.
4.  The chatbot will respond with an answer and the source chunks it used for context.

---

## 📁 Project Structure

```
/agentic-rag-chatbot-gemini
|-- /agents
|   |-- __init__.py
|   |-- base_agent.py         # Abstract base class for all agents
|   |-- ingestion_agent.py    # Agent for parsing and chunking documents
|   |-- retrieval_agent.py    # Agent for embedding and retrieval (Gemini)
|   |-- response_agent.py     # Agent for generating the final LLM response (Gemini)
|-- /utils
|   |-- __init__.py
|   |-- document_parser.py    # Utility functions to parse different file formats
|   |-- mcp.py                # Pydantic models for the Model Context Protocol
|-- app.py                    # Main Streamlit application file and Coordinator logic
|-- requirements.txt          # Python dependencies
|-- .env                      # For API keys (not committed to Git)
|-- README.md                 # This file
```