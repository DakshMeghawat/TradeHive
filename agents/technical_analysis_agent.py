import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any
import ta
from .base_agent import BaseAgent

class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "TechnicalAnalysis")
        self.analysis_cache = {}

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for the given data"""
        try:
            # For single data points, we can't calculate most indicators
            # So we'll generate simplified indicators
            if len(data) <= 1:
                self.logger.info("Single data point received, using simplified indicators")
                close_price = data['Close'].iloc[-1]
                
                # Generate simplified indicators
                indicators = {
                    # Use default values for trend indicators
                    'sma_20': close_price,
                    'ema_20': close_price,
                    'macd': 0.01,  # Slightly positive
                    
                    # Use default values for momentum indicators
                    'rsi': 50.0,  # Neutral
                    'stoch': 50.0,  # Neutral
                    
                    # Use price +/- 10% for Bollinger bands
                    'bollinger_hband': close_price * 1.1,
                    'bollinger_lband': close_price * 0.9,
                    'atr': close_price * 0.02,  # 2% of price
                    
                    # Volume indicator
                    'volume_ema': data['Volume'].iloc[-1]
                }
                
                # Add signal analysis
                indicators['signals'] = self._generate_signals(indicators, close_price)
                
                return indicators
            
            # If we have enough data, calculate real indicators
            # Initialize Technical Analysis indicators
            close_prices = data['Close']
            high_prices = data['High']
            low_prices = data['Low']
            volume = data['Volume']

            indicators = {
                # Trend Indicators
                'sma_20': ta.trend.sma_indicator(close_prices, window=20).iloc[-1],
                'ema_20': ta.trend.ema_indicator(close_prices, window=20).iloc[-1],
                'macd': ta.trend.macd_diff(close_prices).iloc[-1],
                
                # Momentum Indicators
                'rsi': ta.momentum.rsi(close_prices, window=14).iloc[-1],
                'stoch': ta.momentum.stoch(high_prices, low_prices, close_prices).iloc[-1],
                
                # Volatility Indicators
                'bollinger_hband': ta.volatility.bollinger_hband(close_prices).iloc[-1],
                'bollinger_lband': ta.volatility.bollinger_lband(close_prices).iloc[-1],
                'atr': ta.volatility.average_true_range(high_prices, low_prices, close_prices).iloc[-1],
                
                # Volume Indicators
                'volume_ema': ta.volume.volume_weighted_average_price(high_prices, low_prices, close_prices, volume).iloc[-1]
            }

            # Add signal analysis
            indicators['signals'] = self._generate_signals(indicators, close_prices.iloc[-1])
            
            return indicators
        
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return None

    def _generate_signals(self, indicators: Dict[str, float], current_price: float) -> Dict[str, str]:
        """Generate trading signals based on technical indicators"""
        signals = {}
        
        # MACD Signal
        signals['macd'] = 'BUY' if indicators['macd'] > 0 else 'SELL'
        
        # RSI Signal
        if indicators['rsi'] < 30:
            signals['rsi'] = 'BUY'
        elif indicators['rsi'] > 70:
            signals['rsi'] = 'SELL'
        else:
            signals['rsi'] = 'NEUTRAL'
        
        # Bollinger Bands Signal
        if current_price < indicators['bollinger_lband']:
            signals['bollinger'] = 'BUY'
        elif current_price > indicators['bollinger_hband']:
            signals['bollinger'] = 'SELL'
        else:
            signals['bollinger'] = 'NEUTRAL'
        
        # Overall Signal
        buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
        
        if buy_signals > sell_signals:
            signals['overall'] = 'BUY'
        elif sell_signals > buy_signals:
            signals['overall'] = 'SELL'
        else:
            signals['overall'] = 'NEUTRAL'
        
        return signals

    async def process_message(self, message: Dict[str, Any]):
        """Process incoming messages"""
        content = message["content"]
        
        if content.get("type") == "market_data_update":
            market_data = content["data"]
            symbol = market_data["symbol"]
            
            # Skip processing if market data is incomplete
            if market_data.get("current_price") is None:
                self.logger.warning(f"Incomplete market data for {symbol}, skipping technical analysis")
                return
            
            # Check if we have all required data for technical analysis
            required_fields = ["current_price", "volume"]
            additional_fields = ["52w_high", "52w_low"]
            
            if not all(market_data.get(field) is not None for field in required_fields):
                self.logger.warning(f"Missing required data for {symbol}, skipping technical analysis")
                return
                
            if not all(market_data.get("additional_info", {}).get(field) is not None for field in additional_fields):
                self.logger.warning(f"Missing additional data for {symbol}, using default values")
                # Use current price as fallback for high/low if missing
                high = market_data.get("additional_info", {}).get("52w_high", market_data["current_price"] * 1.1)
                low = market_data.get("additional_info", {}).get("52w_low", market_data["current_price"] * 0.9)
            else:
                high = market_data["additional_info"]["52w_high"]
                low = market_data["additional_info"]["52w_low"]
            
            # Convert market data to DataFrame for technical analysis
            df = pd.DataFrame({
                'Close': [market_data['current_price']],
                'High': [high],
                'Low': [low],
                'Volume': [market_data['volume']]
            })
            
            # Calculate technical indicators
            analysis = self.calculate_technical_indicators(df)
            if analysis:
                self.analysis_cache[symbol] = {
                    'timestamp': market_data['timestamp'],
                    'analysis': analysis
                }
                
                # Send analysis to Decision Making Agent
                await self.send_message(
                    "decision_maker_1",
                    {
                        "type": "technical_analysis_update",
                        "symbol": symbol,
                        "data": analysis
                    }
                )

    async def run(self):
        """Main agent loop"""
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
                self.logger.error(f"Error in technical analysis agent main loop: {str(e)}")
                await asyncio.sleep(5) 