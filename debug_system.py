import asyncio
import logging
import json
from datetime import datetime
import os

from agents.base_agent import BaseAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.decision_maker_agent import DecisionMakerAgent
from agents.communication_hub_agent import CommunicationHubAgent

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DebugSystem")

# Mock Market Data Agent that generates predictable data
class MockMarketDataAgent(BaseAgent):
    def __init__(self, agent_id: str, symbols):
        super().__init__(agent_id, "MockMarketData")
        self.symbols = symbols
        self.data_cache = {}
        
    async def generate_mock_data(self, symbol):
        """Generate mock market data with predictable values"""
        mock_data = {
            "symbol": symbol,
            "current_price": 150.0 if symbol == "AAPL" else 300.0,
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat(),
            "daily_change": 0.02,
            "additional_info": {
                "market_cap": 2500000000000,
                "pe_ratio": 25.5,
                "52w_high": 180.0 if symbol == "AAPL" else 350.0,
                "52w_low": 120.0 if symbol == "AAPL" else 250.0
            }
        }
        return mock_data
    
    async def run(self):
        """Main agent loop"""
        self.logger.info(f"Mock Market Data agent started")
        
        # Send initial data immediately
        for symbol in self.symbols:
            data = await self.generate_mock_data(symbol)
            self.data_cache[symbol] = data
            
            # Log the data being sent
            self.logger.debug(f"Sending mock data for {symbol}: {json.dumps(data, indent=2)}")
            
            # Broadcast to Technical Analysis Agent
            await self.send_message(
                "tech_analysis_1",
                {
                    "type": "market_data_update",
                    "data": data
                }
            )
            self.logger.info(f"Sent mock data for {symbol} to tech_analysis_1")
        
        # Keep the agent running but don't send more data
        while self.running:
            await asyncio.sleep(10)

    async def process_message(self, message):
        """Process incoming messages"""
        self.logger.debug(f"Received message: {json.dumps(message, indent=2)}")
        # We don't need to process any messages in this debug agent

async def debug_system():
    """Run a debug version of the trading system with detailed logging"""
    logger.info("Starting debug trading system...")
    
    # Create agents
    symbols = ["AAPL", "MSFT"]
    
    # Create Communication Hub
    comm_hub = CommunicationHubAgent("comm_hub_1")
    
    # Create Mock Market Data Agent
    market_data = MockMarketDataAgent("market_data_1", symbols)
    
    # Create Technical Analysis Agent
    tech_analysis = TechnicalAnalysisAgent("tech_analysis_1")
    
    # Create Decision Maker Agent
    decision_maker = DecisionMakerAgent("decision_maker_1", 0.6)
    
    # Set up subscriptions
    await comm_hub.subscribe("tech_analysis_1", ["market_data_update"])
    await comm_hub.subscribe("decision_maker_1", ["technical_analysis_update"])
    await comm_hub.subscribe("market_data_1", ["trading_decision"])
    
    # Create reports directory if it doesn't exist
    if not os.path.exists("reports"):
        os.makedirs("reports")
    
    # Start all agents
    agents = [comm_hub, market_data, tech_analysis, decision_maker]
    agent_tasks = []
    
    for agent in agents:
        agent_tasks.append(asyncio.create_task(agent.start()))
        logger.info(f"Started {agent.agent_type} agent: {agent.agent_id}")
        await asyncio.sleep(1)  # Small delay between starting agents
    
    try:
        # Wait for a bit to let the system process
        for i in range(5):
            logger.info(f"System running... ({i+1}/5)")
            
            # Check communication hub state
            state = comm_hub.get_latest_state()
            logger.info(f"Current system state: {json.dumps(state, indent=2)}")
            
            # Save the state to a report file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"reports/debug_report_{timestamp}.txt"
            
            with open(report_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"DEBUG SYSTEM REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"SYSTEM STATE:\n{json.dumps(state, indent=2)}\n\n")
                
                # Add agent message queues info
                f.write("AGENT MESSAGE QUEUES:\n")
                for agent in agents:
                    f.write(f"{agent.agent_type} ({agent.agent_id}): {agent.message_queue.qsize()} messages\n")
            
            await asyncio.sleep(10)
        
    except Exception as e:
        logger.error(f"Error in debug system: {str(e)}")
    finally:
        # Stop all agents
        for agent in agents:
            await agent.stop()
            logger.info(f"Stopped {agent.agent_type} agent: {agent.agent_id}")
        
        # Cancel all tasks
        for task in agent_tasks:
            task.cancel()

if __name__ == "__main__":
    # Run the debug system
    asyncio.run(debug_system()) 