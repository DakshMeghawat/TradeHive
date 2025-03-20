from abc import ABC, abstractmethod
from typing import Dict, Any
import asyncio
from datetime import datetime
import logging

# Global message bus for inter-agent communication
MESSAGE_BUS = {}

class BaseAgent(ABC):
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.message_queue = asyncio.Queue()
        self.running = False
        self._setup_logging()
        
        # Register agent in the global message bus
        global MESSAGE_BUS
        MESSAGE_BUS[agent_id] = self.message_queue
        self.logger.debug(f"Agent {agent_id} registered in message bus")

    def _setup_logging(self):
        self.logger = logging.getLogger(f"{self.agent_type}_{self.agent_id}")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def send_message(self, target_agent_id: str, message: Dict[str, Any]):
        """Send a message to another agent"""
        formatted_message = {
            'timestamp': datetime.utcnow().isoformat(),
            'from_agent': self.agent_id,
            'to_agent': target_agent_id,
            'content': message
        }
        
        # Use the global message bus to route the message
        global MESSAGE_BUS
        if target_agent_id in MESSAGE_BUS:
            target_queue = MESSAGE_BUS[target_agent_id]
            await target_queue.put(formatted_message)
            self.logger.debug(f"Message sent to {target_agent_id}")
        else:
            self.logger.error(f"Target agent {target_agent_id} not found in message bus")

    async def receive_message(self) -> Dict[str, Any]:
        """Receive a message from the queue"""
        return await self.message_queue.get()

    @abstractmethod
    async def process_message(self, message: Dict[str, Any]):
        """Process incoming messages"""
        pass

    @abstractmethod
    async def run(self):
        """Main agent loop"""
        pass

    async def start(self):
        """Start the agent"""
        self.running = True
        self.logger.info(f"{self.agent_type} agent {self.agent_id} started")
        await self.run()

    async def stop(self):
        """Stop the agent"""
        self.running = False
        self.logger.info(f"{self.agent_type} agent {self.agent_id} stopped")
        
        # Unregister from message bus
        global MESSAGE_BUS
        if self.agent_id in MESSAGE_BUS:
            del MESSAGE_BUS[self.agent_id]
            self.logger.debug(f"Agent {self.agent_id} unregistered from message bus") 