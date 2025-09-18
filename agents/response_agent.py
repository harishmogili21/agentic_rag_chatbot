import os
import google.generativeai as genai
from .base_agent import Agent
from utils.mcp import MCPMessage

class LLMResponseAgent(Agent):
    """
    Agent responsible for generating a final, user-facing answer by
    calling the Gemini LLM with the user's query and the retrieved context.
    """
    def __init__(self, coordinator_callback):
        super().__init__("LLMResponseAgent", coordinator_callback)
        try:
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        except Exception as e:
            raise Exception("GOOGLE_API_KEY not found. Please set it in your .env file.") from e
        
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def process_message(self, message: MCPMessage):
        if message.type == "GENERATE_REQUEST":
            print(f"[{self.name}] Received GENERATE_REQUEST.")
            query = message.payload.get("query")
            # Accept both context_chunks and retrieved_context for compatibility
            context_chunks = message.payload.get("context_chunks")
            if context_chunks is None:
                context_chunks = message.payload.get("retrieved_context", [])
            trace_id = message.payload.get("trace_id")

            if not context_chunks:
                answer = "I'm sorry, I couldn't find any relevant information in the uploaded documents to answer your question."
                sources = []
            else:
                prompt = self._create_prompt(query, context_chunks)
                #print(f"[LLMResponseAgent] Prompt sent to LLM:\n{prompt}")
                try:
                    response = self.model.generate_content(prompt)
                    answer = response.text
                    sources = context_chunks
                except Exception as e:
                    answer = f"An error occurred while contacting the Gemini API: {e}"
                    sources = []

            response_msg = MCPMessage(
                sender=self.name,
                receiver="Coordinator",
                type="GENERATE_RESPONSE",
                payload={"answer": answer, "sources": sources, "trace_id": trace_id}
            )
            self.send_message(response_msg)
            

    def _create_prompt(self, query: str, context_chunks: list[str]) -> str:
        context_str = "\n\n---\n\n".join(context_chunks)
        prompt = f"""You are a helpful assistant. Answer the user's question based only on the provided context. If the answer is not in the context, say you don't know.

        CONTEXT:
        {context_str}

        QUESTION:
        {query}

        ANSWER:
        """
        return prompt