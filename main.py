import asyncio
import logging
from typing import List, Dict
import json
import time
import os
from dotenv import load_dotenv
from agents.market_data_agent import MarketDataAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.decision_maker_agent import DecisionMakerAgent
from agents.communication_hub_agent import CommunicationHubAgent

# Load environment variables from .env file if it exists
load_dotenv()

class TradingSystem:
    def __init__(self, symbols: List[str], risk_tolerance: float = 0.5, update_interval: int = 60):
        self.symbols = symbols
        self.risk_tolerance = risk_tolerance
        self.update_interval = update_interval
        self.agents = {}
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TradingSystem")

    async def initialize_agents(self):
        """Initialize all trading agents"""
        # Create Communication Hub
        comm_hub = CommunicationHubAgent("comm_hub_1")
        self.agents["comm_hub"] = comm_hub

        # Create Market Data Agent
        market_data = MarketDataAgent("market_data_1", self.symbols, self.update_interval)
        self.agents["market_data"] = market_data

        # Create Technical Analysis Agent
        tech_analysis = TechnicalAnalysisAgent("tech_analysis_1")
        self.agents["tech_analysis"] = tech_analysis

        # Create Decision Maker Agent
        decision_maker = DecisionMakerAgent("decision_maker_1", self.risk_tolerance)
        self.agents["decision_maker"] = decision_maker

        # Set up subscriptions
        await comm_hub.subscribe("tech_analysis_1", ["market_data_update"])
        await comm_hub.subscribe("decision_maker_1", ["technical_analysis_update"])
        await comm_hub.subscribe("market_data_1", ["trading_decision"])

    async def start(self):
        """Start the trading system"""
        self.logger.info("Starting trading system...")
        await self.initialize_agents()

        # Start all agents with delays between them
        agent_tasks = []
        
        # Start communication hub first
        self.logger.info(f"Starting agent: comm_hub")
        agent_tasks.append(asyncio.create_task(self.agents["comm_hub"].start()))
        await asyncio.sleep(2)  # Wait for comm hub to initialize
        
        # Start market data agent
        self.logger.info(f"Starting agent: market_data")
        agent_tasks.append(asyncio.create_task(self.agents["market_data"].start()))
        await asyncio.sleep(2)  # Wait for market data agent to initialize
        
        # Start remaining agents
        for agent_name, agent in self.agents.items():
            if agent_name not in ["comm_hub", "market_data"]:
                self.logger.info(f"Starting agent: {agent_name}")
                agent_tasks.append(asyncio.create_task(agent.start()))
                await asyncio.sleep(1)  # Small delay between starting agents

        try:
            # Wait for all agents to complete (they run indefinitely)
            await asyncio.gather(*agent_tasks)
        except Exception as e:
            self.logger.error(f"Error in trading system: {str(e)}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the trading system"""
        self.logger.info("Stopping trading system...")
        for agent_name, agent in self.agents.items():
            self.logger.info(f"Stopping agent: {agent_name}")
            await agent.stop()

    def get_system_state(self) -> Dict:
        """Get the current state of the trading system"""
        if "comm_hub" in self.agents:
            return self.agents["comm_hub"].get_latest_state()
        return {}

async def main():
    # Get configuration from environment variables or use defaults
    risk_tolerance = float(os.environ.get("RISK_TOLERANCE", 0.6))
    update_interval = int(os.environ.get("UPDATE_INTERVAL", 60))
    
    # Configure the trading system with fewer symbols to avoid rate limiting
    symbols = ["AAPL", "MSFT"]  # Reduced number of symbols
    
    # Log configuration
    logging.info(f"Starting with configuration: symbols={symbols}, risk_tolerance={risk_tolerance}, update_interval={update_interval}")
    
    # Check if Alpha Vantage API key is set
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not api_key or api_key == "demo" or api_key == "your_api_key_here":
        logging.warning("No Alpha Vantage API key found. The system will use mock data.")
        logging.warning("For real market data, get a free API key at: https://www.alphavantage.co/support/#api-key")
        logging.warning("Then set it as an environment variable or in a .env file.")

    # Create and start the trading system
    trading_system = TradingSystem(symbols, risk_tolerance, update_interval)
    
    try:
        # Start the system
        await trading_system.start()
    except KeyboardInterrupt:
        # Handle graceful shutdown
        await trading_system.stop()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 