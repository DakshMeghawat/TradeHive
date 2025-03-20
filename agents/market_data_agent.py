import requests
import pandas as pd
from typing import List, Dict, Any
import asyncio
import random
import time
import os
from datetime import datetime, timedelta
from .base_agent import BaseAgent

# Default Alpha Vantage API key - replace with your own
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")

class MarketDataAgent(BaseAgent):
    def __init__(self, agent_id: str, symbols: List[str], update_interval: int = 60):
        super().__init__(agent_id, "MarketData")
        self.symbols = symbols
        self.update_interval = update_interval
        self.data_cache = {}
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.use_mock_data = self.api_key == "demo"  # Use mock data if no API key
        
        if self.use_mock_data:
            self.logger.warning("No Alpha Vantage API key provided. Using mock data.")

    async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch market data for a given symbol with retry logic"""
        if self.use_mock_data:
            return await self._generate_mock_data(symbol)
            
        retries = 0
        while retries < self.max_retries:
            try:
                self.logger.info(f"Fetching data for {symbol}, attempt {retries+1}")
                
                # Add random delay to avoid rate limiting
                await asyncio.sleep(random.uniform(1.0, 3.0))
                
                # Use Alpha Vantage API instead of Yahoo Finance
                quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
                overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.api_key}"
                
                # Get current quote data
                quote_response = requests.get(quote_url)
                quote_data = quote_response.json()
                
                # Check for API errors
                if "Error Message" in quote_data:
                    raise Exception(f"Alpha Vantage API error: {quote_data['Error Message']}")
                
                # Get company overview data
                await asyncio.sleep(random.uniform(1.0, 2.0))  # Delay between requests
                overview_response = requests.get(overview_url)
                overview_data = overview_response.json()
                
                # Extract relevant data
                global_quote = quote_data.get("Global Quote", {})
                
                if not global_quote:
                    self.logger.warning(f"No quote data found for {symbol}")
                    retries += 1
                    await asyncio.sleep(self.retry_delay)
                    continue
                
                # Format the data
                current_price = float(global_quote.get("05. price", 0))
                market_data = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "volume": int(global_quote.get("06. volume", 0)),
                    "timestamp": datetime.utcnow().isoformat(),
                    "daily_change": float(global_quote.get("10. change percent", "0%").replace("%", "")) / 100,
                    "additional_info": {
                        "market_cap": float(overview_data.get("MarketCapitalization", 0)),
                        "pe_ratio": float(overview_data.get("PERatio", 0)),
                        "52w_high": float(overview_data.get("52WeekHigh", current_price * 1.2)),
                        "52w_low": float(overview_data.get("52WeekLow", current_price * 0.8))
                    }
                }
                
                # Cache the data
                self.data_cache[symbol] = market_data
                return market_data
            
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    # Exponential backoff
                    wait_time = self.retry_delay * (2 ** retries)
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Max retries reached for {symbol}")
                    # Return cached data if available
                    if symbol in self.data_cache:
                        self.logger.info(f"Using cached data for {symbol}")
                        return self.data_cache[symbol]
                    # Fall back to mock data
                    self.logger.info(f"Falling back to mock data for {symbol}")
                    return await self._generate_mock_data(symbol)
        
        return await self._generate_mock_data(symbol)
    
    async def _generate_mock_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock market data when API fails"""
        self.logger.info(f"Generating mock data for {symbol}")
        
        # Base prices for common stocks
        base_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3000.0,
            "TSLA": 700.0
        }
        
        # Use base price or generate a random one
        base_price = base_prices.get(symbol, random.uniform(50.0, 500.0))
        
        # Add some randomness to make it look realistic
        current_price = base_price * random.uniform(0.98, 1.02)
        daily_change = random.uniform(-0.03, 0.03)
        
        mock_data = {
            "symbol": symbol,
            "current_price": current_price,
            "volume": int(random.uniform(500000, 5000000)),
            "timestamp": datetime.utcnow().isoformat(),
            "daily_change": daily_change,
            "additional_info": {
                "market_cap": current_price * random.uniform(1000000000, 3000000000),
                "pe_ratio": random.uniform(15.0, 30.0),
                "52w_high": current_price * random.uniform(1.1, 1.3),
                "52w_low": current_price * random.uniform(0.7, 0.9)
            },
            "is_mock": True
        }
        
        return mock_data

    async def process_message(self, message: Dict[str, Any]):
        """Process incoming messages"""
        content = message["content"]
        if content.get("type") == "data_request":
            symbols = content.get("symbols", self.symbols)
            data = {}
            for symbol in symbols:
                data[symbol] = await self.fetch_market_data(symbol)
            
            await self.send_message(
                message["from_agent"],
                {
                    "type": "market_data_response",
                    "data": data
                }
            )

    async def run(self):
        """Main agent loop"""
        while self.running:
            try:
                # Regular data collection
                for symbol in self.symbols:
                    data = await self.fetch_market_data(symbol)
                    if data:
                        self.data_cache[symbol] = data
                        # Broadcast to Technical Analysis Agent
                        await self.send_message(
                            "tech_analysis_1",
                            {
                                "type": "market_data_update",
                                "data": data
                            }
                        )
                
                # Process any incoming messages
                while not self.message_queue.empty():
                    message = await self.receive_message()
                    await self.process_message(message)
                
                await asyncio.sleep(self.update_interval)
            
            except Exception as e:
                self.logger.error(f"Error in market data agent main loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying 