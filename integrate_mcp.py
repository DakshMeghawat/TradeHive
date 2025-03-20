#!/usr/bin/env python
"""
MCP LLM Integration Script for TradeHive

This script demonstrates how to integrate Multi-Context Protocol (MCP) 
Large Language Models into the TradeHive trading system.
"""

import os
import sys
import asyncio
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Import agent classes
from agents.base_agent import BaseAgent
from agents.market_data_agent import MarketDataAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.decision_maker_agent import DecisionMakerAgent
from agents.communication_hub_agent import CommunicationHubAgent
from mcp_ai_agent import MCPDecisionAgent, create_mcp_agent

# Load environment variables
load_dotenv()

# MCP settings
MCP_ENABLED = os.getenv("MCP_ENABLED", "false").lower() == "true"
MCP_API_KEY = os.getenv("MCP_API_KEY", "")
MCP_API_ENDPOINT = os.getenv("MCP_API_ENDPOINT", "")
MCP_MODEL = os.getenv("MCP_MODEL", "")

# Trading system settings
RISK_TOLERANCE = float(os.getenv("RISK_TOLERANCE", "0.5"))
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "60"))
MONITORED_SYMBOLS = os.getenv("MONITORED_SYMBOLS", "AAPL,MSFT").split(",")

class TradingSystem:
    """Trading system with MCP integration"""
    
    def __init__(self, symbols, use_mcp=False, use_mock_data=False, risk_tolerance=0.5):
        self.symbols = symbols
        self.use_mcp = use_mcp
        self.use_mock_data = use_mock_data
        self.risk_tolerance = risk_tolerance
        self.agents = {}
        self.trading_decisions = {}
        self.mcp_decisions = {}
        
    async def initialize(self):
        """Initialize the trading system"""
        # Create communication hub
        self.agents["comm_hub"] = CommunicationHubAgent("comm_hub_1")
        
        # Create market data agent
        self.agents["market_data"] = MarketDataAgent("market_data_1", use_mock_data=self.use_mock_data)
        
        # Create technical analysis agent
        self.agents["tech_analysis"] = TechnicalAnalysisAgent("tech_analysis_1")
        
        # Create decision maker agent
        self.agents["decision_maker"] = DecisionMakerAgent("decision_maker_1", self.risk_tolerance)
        
        # Create MCP agent if enabled
        if self.use_mcp:
            self.agents["mcp_agent"] = create_mcp_agent("mcp_decision_1", self.risk_tolerance)
        
        # Setup communication hub subscriptions
        comm_hub = self.agents["comm_hub"]
        await comm_hub.add_subscription("tech_analysis_1", "trading_update")
        await comm_hub.add_subscription("decision_maker_1", "technical_analysis_update")
        
        if self.use_mcp:
            await comm_hub.add_subscription("mcp_decision_1", "technical_analysis_update")
        
        # Start all agents
        for agent_id, agent in self.agents.items():
            await agent.start()
            
        print(f"Initialized trading system with {len(self.agents)} agents")
        print(f"MCP integration is {'enabled' if self.use_mcp else 'disabled'}")
        print(f"Monitoring symbols: {', '.join(self.symbols)}")
        
    async def run(self, duration=60):
        """Run the trading system for a specified duration (in seconds)"""
        try:
            # Request market data for monitored symbols
            market_data_agent = self.agents["market_data"]
            for symbol in self.symbols:
                await market_data_agent.send_message(
                    "comm_hub_1",
                    {
                        "type": "request_data",
                        "symbol": symbol
                    }
                )
                
            # Run for the specified duration
            print(f"Running trading system for {duration} seconds...")
            await asyncio.sleep(duration)
                
        except Exception as e:
            print(f"Error running trading system: {str(e)}")
        finally:
            # Stop all agents
            for agent_id, agent in self.agents.items():
                await agent.stop()
                
    async def compare_decisions(self):
        """Compare traditional and MCP-enhanced trading decisions"""
        if not self.use_mcp:
            print("MCP is not enabled, cannot compare decisions")
            return
            
        comm_hub = self.agents["comm_hub"]
        
        # Get decisions from state
        trading_state = comm_hub.get_state()
        decisions = trading_state.get("decisions", {})
        
        # Extract traditional and MCP decisions
        traditional_decisions = []
        mcp_decisions = []
        
        for symbol, decision_data in decisions.items():
            # Get latest traditional decision
            trad_decision = None
            for decision in decision_data.get("trading_decisions", []):
                if "mcp_enhanced" not in decision.get("reasoning", {}):
                    trad_decision = decision
                    break
                    
            # Get latest MCP decision
            mcp_decision = None
            for decision in decision_data.get("mcp_trading_decisions", []):
                if decision.get("reasoning", {}).get("mcp_enhanced", False):
                    mcp_decision = decision
                    break
                    
            if trad_decision and mcp_decision:
                traditional_decisions.append(trad_decision)
                mcp_decisions.append(mcp_decision)
                
        # Print comparison
        print("\n" + "="*80)
        print("TRADING DECISION COMPARISON: TRADITIONAL VS MCP-ENHANCED")
        print("="*80)
        
        for i in range(len(traditional_decisions)):
            trad = traditional_decisions[i]
            mcp = mcp_decisions[i]
            
            symbol = trad["symbol"]
            
            print(f"\nSymbol: {symbol}")
            print("-" * 40)
            
            print(f"{'METRIC':<20} {'TRADITIONAL':<15} {'MCP-ENHANCED':<15} {'DIFFERENCE':<10}")
            print("-" * 80)
            
            # Compare recommendation
            print(f"{'Recommendation':<20} {trad['recommendation']:<15} {mcp['recommendation']:<15} {'MATCH' if trad['recommendation'] == mcp['recommendation'] else 'DIFFERENT':<10}")
            
            # Compare confidence
            trad_conf = trad["confidence_score"]
            mcp_conf = mcp["confidence_score"]
            conf_diff = abs(mcp_conf - trad_conf)
            print(f"{'Confidence Score':<20} {trad_conf:.2f}{'':>8} {mcp_conf:.2f}{'':>8} {conf_diff:.2f}{'':>4}")
            
            # Compare position size
            trad_pos = trad["suggested_position_size"]
            mcp_pos = mcp["suggested_position_size"]
            pos_diff = abs(mcp_pos - trad_pos)
            print(f"{'Position Size':<20} {trad_pos:.2f}%{'':>7} {mcp_pos:.2f}%{'':>7} {pos_diff:.2f}%{'':>3}")
            
            # Show MCP-specific insights
            print("\nMCP-Enhanced Insights:")
            print(f"  Rationale: {mcp['reasoning'].get('rationale', 'N/A')}")
            print(f"  Risks: {', '.join(mcp['reasoning'].get('risks', ['N/A']))}")
            print(f"  Time Horizon: {mcp['reasoning'].get('time_horizon', 'N/A')}")
            
            print("-" * 80)
            
        print("\n" + "="*80)
        
    def generate_report(self, output_file="mcp_comparison_report.json"):
        """Generate a report comparing traditional and MCP decisions"""
        if not self.use_mcp:
            print("MCP is not enabled, cannot generate comparison report")
            return
            
        comm_hub = self.agents["comm_hub"]
        
        # Get decisions from state
        trading_state = comm_hub.get_state()
        decisions = trading_state.get("decisions", {})
        
        # Prepare report data
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbols": self.symbols,
            "mcp_enabled": self.use_mcp,
            "risk_tolerance": self.risk_tolerance,
            "comparisons": []
        }
        
        # Extract and compare decisions
        for symbol, decision_data in decisions.items():
            # Get latest traditional decision
            trad_decision = None
            for decision in decision_data.get("trading_decisions", []):
                if "mcp_enhanced" not in decision.get("reasoning", {}):
                    trad_decision = decision
                    break
                    
            # Get latest MCP decision
            mcp_decision = None
            for decision in decision_data.get("mcp_trading_decisions", []):
                if decision.get("reasoning", {}).get("mcp_enhanced", False):
                    mcp_decision = decision
                    break
                    
            if trad_decision and mcp_decision:
                comparison = {
                    "symbol": symbol,
                    "traditional_decision": trad_decision,
                    "mcp_decision": mcp_decision,
                    "comparison": {
                        "recommendation_match": trad_decision["recommendation"] == mcp_decision["recommendation"],
                        "confidence_difference": abs(mcp_decision["confidence_score"] - trad_decision["confidence_score"]),
                        "position_size_difference": abs(mcp_decision["suggested_position_size"] - trad_decision["suggested_position_size"])
                    }
                }
                report_data["comparisons"].append(comparison)
        
        # Save report to file
        os.makedirs("reports", exist_ok=True)
        report_path = os.path.join("reports", output_file)
        
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        print(f"Generated MCP comparison report: {report_path}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run TradeHive with MCP integration')
    parser.add_argument('--symbols', type=str, default=','.join(MONITORED_SYMBOLS),
                        help='Comma-separated list of stock symbols to monitor')
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration to run the system in seconds')
    parser.add_argument('--use-mcp', action='store_true', default=MCP_ENABLED,
                        help='Enable MCP integration')
    parser.add_argument('--mock-data', action='store_true', default=USE_MOCK_DATA,
                        help='Use mock market data')
    parser.add_argument('--risk', type=float, default=RISK_TOLERANCE,
                        help='Risk tolerance (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Check if MCP is enabled but API key is missing
    if args.use_mcp and not MCP_API_KEY:
        print("Warning: MCP is enabled but no API key is provided in .env file")
        print("Continuing with mock MCP responses")
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Create and run trading system
    system = TradingSystem(
        symbols=symbols,
        use_mcp=args.use_mcp,
        use_mock_data=args.mock_data,
        risk_tolerance=args.risk
    )
    
    await system.initialize()
    await system.run(args.duration)
    
    # Compare decisions and generate report
    if args.use_mcp:
        await system.compare_decisions()
        system.generate_report()

if __name__ == "__main__":
    asyncio.run(main()) 