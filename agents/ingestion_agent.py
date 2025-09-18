from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base_agent import Agent
from utils.mcp import MCPMessage
from utils.document_parser import parse_document
import os

class IngestionAgent(Agent):
    """
    Agent responsible for parsing documents, splitting them into chunks,
    and sending them for embedding.
    """
    def __init__(self, coordinator_callback):
        super().__init__("IngestionAgent", coordinator_callback)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_message(self, message: MCPMessage):
        if message.type == "INGEST_REQUEST":
            print(f"[{self.name}] Received INGEST_REQUEST.")
            file_paths = message.payload["file_paths"]
            all_chunks = []
            all_metadata = []

            for path in file_paths:
                try:
                    content = parse_document(path)
                    chunks = self.text_splitter.split_text(content)
                    metadata = [{"source": os.path.basename(path)} for _ in chunks]
                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)
                    print(f"[{self.name}] Parsed and chunked {os.path.basename(path)}.")
                except Exception as e:
                    print(f"[{self.name}] Error parsing {path}: {e}")

            if all_chunks:
                response_msg = MCPMessage(
                    sender=self.name,
                    receiver="Coordinator",
                    type="EMBED_REQUEST",
                    payload={"chunks": all_chunks, "metadata": all_metadata}
                )
                self.send_message(response_msg)