import asyncio
import logging
import json
from datetime import datetime
from agents.base_agent import BaseAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.decision_maker_agent import DecisionMakerAgent
from agents.communication_hub_agent import CommunicationHubAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestSystem")

# Mock Market Data Agent that doesn't use yfinance
class MockMarketDataAgent(BaseAgent):
    def __init__(self, agent_id: str, symbols):
        super().__init__(agent_id, "MockMarketData")
        self.symbols = symbols
        self.data_cache = {}
        
    async def generate_mock_data(self, symbol):
        """Generate mock market data instead of fetching from API"""
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
        
    async def process_message(self, message):
        """Process incoming messages"""
        content = message["content"]
        if content.get("type") == "data_request":
            symbols = content.get("symbols", self.symbols)
            data = {}
            for symbol in symbols:
                data[symbol] = await self.generate_mock_data(symbol)
            
            await self.send_message(
                message["from_agent"],
                {
                    "type": "market_data_response",
                    "data": data
                }
            )
    
    async def run(self):
        """Main agent loop"""
        self.logger.info(f"Mock Market Data agent started")
        while self.running:
            try:
                # Generate mock data for each symbol
                for symbol in self.symbols:
                    data = await self.generate_mock_data(symbol)
                    self.data_cache[symbol] = data
                    
                    # Broadcast to Technical Analysis Agent
                    await self.send_message(
                        "tech_analysis_1",
                        {
                            "type": "market_data_update",
                            "data": data
                        }
                    )
                    self.logger.info(f"Sent mock data for {symbol}")
                    
                # Process any incoming messages
                while not self.message_queue.empty():
                    message = await self.receive_message()
                    await self.process_message(message)
                
                # Wait longer between updates for testing
                await asyncio.sleep(10)
            
            except Exception as e:
                self.logger.error(f"Error in mock market data agent: {str(e)}")
                await asyncio.sleep(5)

async def run_test():
    """Run a test of the trading system with mock data"""
    logger.info("Starting test trading system...")
    
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
    
    # Start all agents
    agents = [comm_hub, market_data, tech_analysis, decision_maker]
    agent_tasks = []
    
    for agent in agents:
        agent_tasks.append(asyncio.create_task(agent.start()))
        await asyncio.sleep(1)  # Small delay between starting agents
    
    try:
        # Run for a limited time for testing
        await asyncio.sleep(60)  # Run for 60 seconds
        
        # Get final state
        state = comm_hub.get_latest_state()
        logger.info(f"Final system state: {json.dumps(state, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error in test system: {str(e)}")
    finally:
        # Stop all agents
        for agent in agents:
            await agent.stop()
        
        # Cancel all tasks
        for task in agent_tasks:
            task.cancel()

if __name__ == "__main__":
    # Run the test
    asyncio.run(run_test()) 