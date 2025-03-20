import os
import asyncio
import json
import logging
import requests
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from agents.base_agent import BaseAgent

# Load environment variables
load_dotenv()

# MCP LLM API Configuration
MCP_API_KEY = os.getenv("MCP_API_KEY", "")
MCP_API_ENDPOINT = os.getenv("MCP_API_ENDPOINT", "https://api.your-mcp-provider.com/v1")
MCP_MODEL = os.getenv("MCP_MODEL", "mcp-agent-v1")

class MCPDecisionAgent(BaseAgent):
    """
    Multi-Context Protocol LLM agent for enhanced trading decisions.
    
    This agent leverages advanced LLM capabilities to analyze market data
    across multiple contexts and provide more nuanced trading decisions.
    """
    
    def __init__(self, agent_id: str, risk_tolerance: float = 0.5):
        super().__init__(agent_id, "MCP_Decision")
        self.risk_tolerance = risk_tolerance
        self.api_key = MCP_API_KEY
        self.api_endpoint = MCP_API_ENDPOINT
        self.model = MCP_MODEL
        self.market_context = {}
        self.company_profiles = {}
        self.news_sentiment = {}
        self.technical_data = {}
        self.decisions_cache = {}
        self.should_stop = False
        
    async def run(self):
        """Main agent loop"""
        self.logger.info(f"MCP Decision agent {self.agent_id} started")
        
        try:
            while not self.should_stop:
                # Process messages from queue
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self.process_message(message)
                
                # Sleep to avoid high CPU usage
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in MCP Decision agent: {str(e)}")
        finally:
            self.logger.info(f"MCP Decision agent {self.agent_id} stopped")
    
    async def stop(self):
        """Stop the agent"""
        self.should_stop = True
        await asyncio.sleep(0.5)  # Give time for the run loop to exit
    
    async def fetch_company_profile(self, symbol: str) -> Dict:
        """Fetch company profile and fundamentals"""
        try:
            # This would be replaced with actual API call to fetch company data
            # For now, we'll use mock data
            profile = {
                "symbol": symbol,
                "name": f"{symbol} Corporation",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000000000,
                "pe_ratio": 20.5,
                "dividend_yield": 1.2,
                "beta": 1.1,
                "52_week_high": 200.0,
                "52_week_low": 100.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.company_profiles[symbol] = profile
            return profile
        except Exception as e:
            self.logger.error(f"Error fetching company profile for {symbol}: {str(e)}")
            return {}
    
    async def fetch_news_sentiment(self, symbol: str) -> Dict:
        """Fetch recent news and sentiment analysis for the symbol"""
        try:
            # This would be replaced with actual API call to fetch news data
            # For now, we'll use mock data
            sentiment = {
                "symbol": symbol,
                "sentiment_score": 0.65,  # -1 to 1, where 1 is most positive
                "sentiment_magnitude": 0.8,  # 0 to 1, where 1 is stronger sentiment
                "news_count": 5,
                "latest_headlines": [
                    f"{symbol} Reports Strong Quarterly Earnings",
                    f"{symbol} Announces New Product Line",
                    f"Analysts Upgrade {symbol} Stock Rating"
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.news_sentiment[symbol] = sentiment
            return sentiment
        except Exception as e:
            self.logger.error(f"Error fetching news sentiment for {symbol}: {str(e)}")
            return {}
    
    async def fetch_market_context(self) -> Dict:
        """Fetch broader market context and macroeconomic indicators"""
        try:
            # This would be replaced with actual API call to fetch market data
            # For now, we'll use mock data
            context = {
                "sp500_change": 0.5,  # percentage
                "nasdaq_change": 0.7,  # percentage
                "vix": 15.3,  # volatility index
                "treasury_10y": 3.5,  # 10-year treasury yield
                "fed_rate": 5.25,  # Federal Reserve interest rate
                "inflation_rate": 3.2,  # inflation rate
                "gdp_growth": 2.1,  # GDP growth rate
                "unemployment_rate": 3.8,  # unemployment rate
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.market_context = context
            return context
        except Exception as e:
            self.logger.error(f"Error fetching market context: {str(e)}")
            return {}
    
    async def generate_mcp_prompt(self, symbol: str) -> str:
        """Generate a multi-context prompt for the LLM"""
        # Fetch all necessary context if not already available
        if symbol not in self.company_profiles:
            await self.fetch_company_profile(symbol)
        
        if symbol not in self.news_sentiment:
            await self.fetch_news_sentiment(symbol)
        
        if not self.market_context:
            await self.fetch_market_context()
        
        # Get technical analysis data
        technical_data = self.technical_data.get(symbol, {})
        
        # Create the multi-context prompt
        prompt = f"""
# Market Analysis for {symbol}

## Technical Analysis Context
- Current Price: ${technical_data.get('price', 'N/A')}
- Volume: {technical_data.get('volume', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}
- RSI: {technical_data.get('rsi', 'N/A')}
- Bollinger Bands:
  - Upper: {technical_data.get('bb_upper', 'N/A')}
  - Middle: {technical_data.get('bb_middle', 'N/A')}
  - Lower: {technical_data.get('bb_lower', 'N/A')}
- Moving Averages:
  - 50-day MA: {technical_data.get('ma_50', 'N/A')}
  - 200-day MA: {technical_data.get('ma_200', 'N/A')}

## Company Fundamentals Context
- Company: {self.company_profiles.get(symbol, {}).get('name', 'N/A')}
- Sector: {self.company_profiles.get(symbol, {}).get('sector', 'N/A')}
- Industry: {self.company_profiles.get(symbol, {}).get('industry', 'N/A')}
- Market Cap: ${self.company_profiles.get(symbol, {}).get('market_cap', 'N/A')}
- P/E Ratio: {self.company_profiles.get(symbol, {}).get('pe_ratio', 'N/A')}
- Dividend Yield: {self.company_profiles.get(symbol, {}).get('dividend_yield', 'N/A')}%
- 52-Week Range: ${self.company_profiles.get(symbol, {}).get('52_week_low', 'N/A')} - ${self.company_profiles.get(symbol, {}).get('52_week_high', 'N/A')}

## News Sentiment Context
- Overall Sentiment: {self.news_sentiment.get(symbol, {}).get('sentiment_score', 'N/A')} (Scale: -1 to 1)
- News Count: {self.news_sentiment.get(symbol, {}).get('news_count', 'N/A')}
- Recent Headlines:
  {' '.join(['- ' + headline for headline in self.news_sentiment.get(symbol, {}).get('latest_headlines', ['N/A'])])}

## Broader Market Context
- S&P 500 Change: {self.market_context.get('sp500_change', 'N/A')}%
- NASDAQ Change: {self.market_context.get('nasdaq_change', 'N/A')}%
- VIX (Volatility Index): {self.market_context.get('vix', 'N/A')}
- 10-Year Treasury Yield: {self.market_context.get('treasury_10y', 'N/A')}%
- Fed Interest Rate: {self.market_context.get('fed_rate', 'N/A')}%
- Inflation Rate: {self.market_context.get('inflation_rate', 'N/A')}%
- GDP Growth: {self.market_context.get('gdp_growth', 'N/A')}%
- Unemployment Rate: {self.market_context.get('unemployment_rate', 'N/A')}%

Based on these multiple contexts, analyze {symbol} and provide:
1. A trading recommendation (BUY, SELL, or NEUTRAL)
2. A confidence score (0.0 to 1.0)
3. Suggested position size as a percentage of available capital, considering risk tolerance of {self.risk_tolerance} (Scale: 0-1)
4. Rationale for the recommendation
5. Key risks to consider
6. Expected time horizon (short-term, medium-term, long-term)

Format your response as a JSON object with these fields.
"""
        return prompt
    
    async def call_mcp_llm(self, prompt: str) -> Dict:
        """Call the MCP LLM API to get a trading decision"""
        try:
            if not self.api_key:
                self.logger.warning("MCP API key not configured. Using mock response.")
                # Return mock response
                return {
                    "recommendation": "BUY",
                    "confidence_score": 0.75,
                    "suggested_position_size": 0.15,
                    "rationale": "Mock rationale based on technical signals and market conditions",
                    "risks": ["Market volatility", "Sector rotation", "Earnings risk"],
                    "time_horizon": "medium-term"
                }
            
            # Make API call to MCP LLM
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 500,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                f"{self.api_endpoint}/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result.get("choices", [{}])[0].get("message", {}).get("content", "{}"))
            else:
                self.logger.error(f"MCP API error: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error calling MCP LLM: {str(e)}")
            return {}
    
    async def generate_decision(self, symbol: str) -> Dict:
        """Generate a trading decision using MCP LLM"""
        # Generate the prompt with multiple contexts
        prompt = await self.generate_mcp_prompt(symbol)
        
        # Call the MCP LLM
        llm_response = await self.call_mcp_llm(prompt)
        
        if not llm_response:
            self.logger.warning(f"No response from MCP LLM for {symbol}. Using fallback.")
            # Fallback to a simple decision based on technical data
            technical_data = self.technical_data.get(symbol, {})
            signals = technical_data.get('signals', {})
            
            # Basic fallback logic
            recommendation = signals.get('overall', 'NEUTRAL')
            confidence_score = 0.5
            position_size = 0.1 * self.risk_tolerance
            
            decision = {
                "symbol": symbol,
                "recommendation": recommendation,
                "confidence_score": confidence_score,
                "suggested_position_size": position_size,
                "reasoning": {
                    "technical_signals": signals,
                    "fallback_mode": True
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Process LLM response
            decision = {
                "symbol": symbol,
                "recommendation": llm_response.get("recommendation", "NEUTRAL"),
                "confidence_score": float(llm_response.get("confidence_score", 0.5)),
                "suggested_position_size": float(llm_response.get("suggested_position_size", 0.1)),
                "reasoning": {
                    "rationale": llm_response.get("rationale", ""),
                    "risks": llm_response.get("risks", []),
                    "time_horizon": llm_response.get("time_horizon", "medium-term"),
                    "mcp_enhanced": True
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Cache the decision
        self.decisions_cache[symbol] = decision
        
        return decision
    
    async def process_message(self, message: Dict[str, Any]):
        """Process incoming messages from other agents"""
        message_from = message.get("from", "")
        content = message.get("content", {})
        msg_type = content.get("type", "")
        
        if msg_type == "technical_analysis_update":
            symbol = content.get("symbol", "")
            data = content.get("data", {})
            
            # Store technical data
            self.technical_data[symbol] = data
            
            # Generate trading decision with MCP
            decision = await self.generate_decision(symbol)
            
            # Broadcast decision to communication hub
            await self.send_message(
                "comm_hub_1",
                {
                    "type": "mcp_trading_decision",
                    "decision": decision
                }
            )
            
            self.logger.info(f"Sent MCP-enhanced trading decision for {symbol}: {decision['recommendation']}")

# Factory function to create the MCP agent
def create_mcp_agent(agent_id="mcp_decision_1", risk_tolerance=0.5):
    """Create and return an instance of the MCP Decision Agent"""
    return MCPDecisionAgent(agent_id, risk_tolerance) 