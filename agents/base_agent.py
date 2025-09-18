from abc import ABC, abstractmethod
from typing import Callable
from utils.mcp import MCPMessage

class Agent(ABC):
    """Abstract Base Class for all agents."""
    def __init__(self, name: str, coordinator_callback: Callable):
        self.name = name
        self.coordinator_callback = coordinator_callback

    @abstractmethod
    def process_message(self, message: MCPMessage):
        """Each agent must implement this method to handle incoming messages."""
        pass

    def send_message(self, message: MCPMessage):
        """Sends a message back to the coordinator."""
        self.coordinator_callback(message)