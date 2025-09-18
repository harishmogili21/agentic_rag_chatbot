import uuid
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field

# Define literal types for type safety and clarity
SenderType = Literal["UI", "Coordinator", "IngestionAgent", "RetrievalAgent", "LLMResponseAgent"]
ReceiverType = Literal["Coordinator", "IngestionAgent", "RetrievalAgent", "LLMResponseAgent", "UI"]
MessageType = Literal[
    "INGEST_REQUEST",
    "INGEST_COMPLETE",
    "EMBED_REQUEST",
    "EMBED_COMPLETE",
    "RETRIEVAL_REQUEST",
    "RETRIEVAL_RESPONSE",
    "GENERATE_REQUEST",
    "GENERATE_RESPONSE"
]

class MCPPayload(BaseModel):
    """Base model for MCP payloads to allow flexible data structures."""
    pass

class IngestRequestPayload(MCPPayload):
    file_paths: List[str]

class EmbedRequestPayload(MCPPayload):
    chunks: List[str]
    metadata: List[Dict[str, Any]]

class RetrievalRequestPayload(MCPPayload):
    query: str

class RetrievalResponsePayload(MCPPayload):
    query: str
    retrieved_context: List[str]

class GenerateRequestPayload(MCPPayload):
    query: str
    context_chunks: List[str]

class GenerateResponsePayload(MCPPayload):
    answer: str
    sources: List[str]

class MCPMessage(BaseModel):
    """
    Defines the structured message for the Model Context Protocol (MCP).
    """
    sender: SenderType
    receiver: ReceiverType
    type: MessageType
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any]

    def to_dict(self):
        return self.model_dump()