import asyncio
import logging
import json
from datetime import datetime

from agents.base_agent import BaseAgent

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestCommunication")

class SimpleAgent(BaseAgent):
    def __init__(self, agent_id: str, agent_type: str):
        super().__init__(agent_id, agent_type)
        
    async def process_message(self, message):
        """Process incoming messages"""
        self.logger.info(f"Received message: {json.dumps(message, indent=2)}")
        
        # Echo back the message to the sender
        if "from_agent" in message:
            await self.send_message(
                message["from_agent"],
                {
                    "type": "echo_response",
                    "original_message": message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            self.logger.info(f"Sent echo response to {message['from_agent']}")
    
    async def run(self):
        """Main agent loop"""
        self.logger.info(f"{self.agent_type} agent {self.agent_id} running")
        
        while self.running:
            try:
                # Process any incoming messages
                self.logger.debug(f"Checking message queue, size: {self.message_queue.qsize()}")
                while not self.message_queue.empty():
                    message = await self.receive_message()
                    self.logger.debug(f"Processing message: {message}")
                    await self.process_message(message)
                
                await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in agent loop: {str(e)}")
                await asyncio.sleep(1)

async def test_communication():
    """Test direct communication between agents"""
    logger.info("Starting communication test...")
    
    # Create two simple agents
    agent1 = SimpleAgent("agent1", "TestAgent")
    agent2 = SimpleAgent("agent2", "TestAgent")
    
    # Start both agents
    task1 = asyncio.create_task(agent1.start())
    task2 = asyncio.create_task(agent2.start())
    
    # Wait for agents to start
    await asyncio.sleep(1)
    
    try:
        # Send a test message from agent1 to agent2
        logger.info("Sending test message from agent1 to agent2")
        await agent1.send_message(
            "agent2",
            {
                "type": "test_message",
                "content": "Hello from agent1!",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Wait for message processing
        await asyncio.sleep(3)
        
        # Send another test message
        logger.info("Sending test message from agent2 to agent1")
        await agent2.send_message(
            "agent1",
            {
                "type": "test_message",
                "content": "Hello from agent2!",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Wait for message processing
        await asyncio.sleep(3)
        
    finally:
        # Stop the agents
        await agent1.stop()
        await agent2.stop()
        
        # Cancel the tasks
        task1.cancel()
        task2.cancel()

if __name__ == "__main__":
    # Run the communication test
    asyncio.run(test_communication()) 