import faiss
import numpy as np
import google.generativeai as genai
import os
from .base_agent import Agent
from utils.mcp import MCPMessage
from sentence_transformers import SentenceTransformer

class RetrievalAgent(Agent):
    """
    Agent responsible for creating and storing vector embeddings (using Gemini & FAISS)
    and retrieving relevant document chunks based on a user query.
    """
    def __init__(self, coordinator_callback):
        import pickle
        super().__init__("RetrievalAgent", coordinator_callback)
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        except Exception as e:
            raise Exception("GOOGLE_API_KEY not found. Please set it in your .env file.") from e

        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.model = SentenceTransformer(self.embedding_model_name)
        self.index = None
        self.chunks_with_metadata = []

        # Persistence paths
        self.index_path = "faiss_index.bin"
        self.meta_path = "faiss_chunks.pkl"

        # Try to load index and metadata
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.chunks_with_metadata = pickle.load(f)
                print(f"[RetrievalAgent] Loaded FAISS index and metadata from disk.")
            except Exception as e:
                print(f"[RetrievalAgent] Failed to load FAISS index or metadata: {e}")

    def _initialize_index(self, embedding_dim: int):
        if self.index is None:
            # Gemini 'embedding-001' model has a dimension of 768
            self.index = faiss.IndexFlatL2(embedding_dim)

    def process_message(self, message: MCPMessage):
        if message.type == "EMBED_REQUEST":
            import pickle
            print(f"[{self.name}] Received EMBED_REQUEST.")
            chunks = message.payload["chunks"]
            metadata = message.payload["metadata"]

            if not chunks:
                print(f"[{self.name}] No chunks to embed.")
                return

            embeddings = self.model.encode(chunks)
            embedding_dim = len(embeddings[0])
            self._initialize_index(embedding_dim)

            # Store chunks and metadata before adding to index
            self.chunks_with_metadata.extend(list(zip(chunks, metadata)))
            self.index.add(np.array(embeddings).astype('float32'))

            # Save index and metadata to disk
            try:
                faiss.write_index(self.index, self.index_path)
                with open(self.meta_path, "wb") as f:
                    pickle.dump(self.chunks_with_metadata, f)
                print(f"[{self.name}] Saved FAISS index and metadata to disk.")
            except Exception as e:
                print(f"[{self.name}] Failed to save FAISS index or metadata: {e}")

            print(f"[{self.name}] Added {len(chunks)} chunks to FAISS index.")

            # Notify Coordinator of completion
            response_msg = MCPMessage(
                sender=self.name,
                receiver="Coordinator",
                type="INGEST_COMPLETE",
                payload={}
            )
            self.send_message(response_msg)

        elif message.type == "RETRIEVAL_REQUEST":
            print(f"[{self.name}] Received RETRIEVAL_REQUEST.")
            query = message.payload["query"]
            trace_id = getattr(message, "trace_id", None) or message.payload.get("trace_id")
            if not trace_id:
                import uuid
                trace_id = str(uuid.uuid4())

            if self.index is None or self.index.ntotal == 0:
                print(f"[{self.name}] Vector store is not initialized or is empty.")
                context_chunks = []
            else:
                print(f"[{self.name}] Searching FAISS index with {self.index.ntotal} vectors...")
                embeddings = self.model.encode(query)
                #print(f"[{self.name}] Query embedding: {embeddings}")
                query_embedding = np.array([embeddings]).astype('float32')
                k = min(3, self.index.ntotal) # Retrieve top 5 or fewer if not enough docs
                distances, indices = self.index.search(query_embedding, k)
                #print(f"[{self.name}] FAISS search distances: {distances}, indices: {indices}")
                context_chunks = [self.chunks_with_metadata[i][0] for i in indices[0]]
                #print(f"[{self.name}] Retrieved {len(context_chunks)} chunks for query.")

            response_msg = MCPMessage(
                sender=self.name,
                receiver="Coordinator",
                type="RETRIEVAL_RESPONSE",
                trace_id=trace_id,
                payload={
                    "retrieved_context": context_chunks,
                    "query": query,
                    "trace_id": trace_id
                }
            )
            #print(f"[{self.name}] Sending RETRIEVAL_RESPONSE to Coordinator: {response_msg}")
            self.send_message(response_msg)