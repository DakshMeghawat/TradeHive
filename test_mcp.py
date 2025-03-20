#!/usr/bin/env python
"""
Test script for Multi-Context Protocol (MCP) integration
"""

import os
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
from mcp_ai_agent import MCPDecisionAgent

# Mock MCP response for demonstration
MOCK_MCP_RESPONSE = {
    "recommendation": "BUY",
    "confidence_score": 0.78,
    "suggested_position_size": 0.15,
    "rationale": "AAPL shows strong technical signals with MACD and Bollinger Bands indicating upward momentum. The price is above both 50-day and 200-day moving averages, showing a solid uptrend. Fundamentally, the company's strong market position in the technology sector, combined with positive news sentiment (0.65) suggests continued growth potential. The broader market conditions are also favorable with S&P 500 and NASDAQ showing positive momentum (0.5% and 0.7% respectively). The relatively low volatility index (VIX at 15.3) indicates market stability. Considering all contexts, AAPL presents a compelling buying opportunity with a reasonable risk profile.",
    "risks": [
        "Potential market volatility due to macroeconomic uncertainties",
        "Sector rotation away from technology stocks",
        "Earnings expectations may be priced in, creating potential for disappointment",
        "Supply chain disruptions affecting production capacity",
        "Regulatory challenges in key markets"
    ],
    "time_horizon": "medium-term"
}

class EnhancedMCPDecisionAgent(MCPDecisionAgent):
    """Enhanced version of the MCP Decision Agent with better mock responses"""
    
    async def call_mcp_llm(self, prompt: str) -> dict:
        """Override to provide a more sophisticated mock response"""
        try:
            if not self.api_key or "your-mcp-provider.com" in self.api_endpoint:
                self.logger.warning("Using enhanced mock MCP response for demonstration")
                # Return enhanced mock response
                return MOCK_MCP_RESPONSE
            
            # Original implementation for real API calls
            return await super().call_mcp_llm(prompt)
                
        except Exception as e:
            self.logger.error(f"Error calling MCP LLM: {str(e)}")
            return MOCK_MCP_RESPONSE  # Fall back to mock response on error

async def main():
    """Test the MCP agent with a simple example"""
    # Load environment variables
    load_dotenv()
    
    print("=== Multi-Context Protocol LLM Test ===")
    
    # Check environment variables
    mcp_api_key = os.getenv("MCP_API_KEY", "")
    mcp_enabled = os.getenv("MCP_ENABLED", "false").lower() == "true"
    
    print(f"MCP Enabled: {mcp_enabled}")
    if mcp_api_key:
        print(f"MCP API Key: {mcp_api_key[:5]}...{mcp_api_key[-4:] if len(mcp_api_key) > 8 else ''}")
    else:
        print("MCP API Key: Not configured, will use mock responses")
    
    # Create enhanced MCP agent
    print("\nCreating Enhanced MCP Decision Agent...")
    agent = EnhancedMCPDecisionAgent("test_mcp_agent_1", risk_tolerance=0.6)
    
    # Set up test data
    print("\nSetting up test data for AAPL...")
    
    # Add technical data
    agent.technical_data["AAPL"] = {
        "price": 187.50,
        "volume": 25000000,
        "macd": 1.25,
        "rsi": 62,
        "bb_upper": 190.25,
        "bb_middle": 185.50,
        "bb_lower": 180.75,
        "ma_50": 182.30,
        "ma_200": 175.80,
        "signals": {
            "overall": "BUY",
            "macd": "BUY",
            "rsi": "NEUTRAL",
            "bollinger": "BUY"
        }
    }
    
    # Generate and print MCP prompt
    print("\nGenerating MCP Prompt...")
    prompt = await agent.generate_mcp_prompt("AAPL")
    print("\n--- MCP PROMPT ---")
    print(prompt)
    
    # Get MCP decision
    print("\nGenerating MCP Decision...")
    decision = await agent.generate_decision("AAPL")
    
    # Print decision with nice formatting
    print("\n" + "="*80)
    print("MCP TRADING DECISION")
    print("="*80)
    print(f"Symbol: {decision['symbol']}")
    print(f"Recommendation: {decision['recommendation']}")
    print(f"Confidence Score: {decision['confidence_score']:.2f}")
    print(f"Suggested Position Size: {decision['suggested_position_size']:.2f}%")
    
    if "rationale" in decision.get("reasoning", {}):
        print("\nRATIONALE:")
        print("-"*80)
        print(decision['reasoning']['rationale'])
    
    if "risks" in decision.get("reasoning", {}):
        print("\nRISKS:")
        print("-"*80)
        for risk in decision['reasoning'].get('risks', []):
            print(f"• {risk}")
    
    if "time_horizon" in decision.get("reasoning", {}):
        print("\nTIME HORIZON:")
        print("-"*80)
        print(decision['reasoning'].get('time_horizon', 'N/A'))
    
    # Compare with traditional analysis
    print("\n" + "="*80)
    print("COMPARISON WITH TRADITIONAL ANALYSIS")
    print("="*80)
    
    traditional_recommendation = agent.technical_data["AAPL"]["signals"]["overall"]
    confidence_boost = decision['confidence_score'] - 0.5  # Assuming baseline confidence of 0.5
    
    print(f"{'METRIC':<20} {'TRADITIONAL':<15} {'MCP-ENHANCED':<15} {'DIFFERENCE':<10}")
    print("-" * 80)
    print(f"{'Recommendation':<20} {traditional_recommendation:<15} {decision['recommendation']:<15} {'MATCH' if traditional_recommendation == decision['recommendation'] else 'DIFFERENT':<10}")
    print(f"{'Confidence Score':<20} {'0.50':<15} {decision['confidence_score']:.2f}{'':<8} {confidence_boost:.2f}{'':<4}")
    print(f"{'Position Size':<20} {'3.00%':<15} {decision['suggested_position_size']:.2f}%{'':<7} {(decision['suggested_position_size'] - 3.0):.2f}%{'':<3}")
    print(f"{'Context Sources':<20} {'Technical':<15} {'Multi-Context':<15} {'ENHANCED':<10}")
    print(f"{'Time Horizon':<20} {'N/A':<15} {decision['reasoning'].get('time_horizon', 'N/A'):<15} {'ADDED':<10}")
    
    if decision.get("reasoning", {}).get("fallback_mode", False):
        print("\nNote: Using fallback mode (basic response)")
    elif decision.get("reasoning", {}).get("mcp_enhanced", False):
        print("\nNote: Using MCP-enhanced decision with multiple context analysis")
    
    print("\nTest completed successfully.")

if __name__ == "__main__":
    asyncio.run(main()) 