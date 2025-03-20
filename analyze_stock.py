import asyncio
import sys
import json
import os
import random
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from agents.base_agent import BaseAgent
from agents.technical_analysis_agent import TechnicalAnalysisAgent
from agents.decision_maker_agent import DecisionMakerAgent
from agents.communication_hub_agent import CommunicationHubAgent
from agents.market_data_agent import MarketDataAgent

# Load environment variables
load_dotenv()

# Paper trading API keys and settings
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "false").lower() == "true"
PAPER_TRADING_API = os.getenv("PAPER_TRADING_API", "alpaca")  # Options: alpaca, interactive_brokers, etc.
PAPER_TRADING_API_KEY = os.getenv("PAPER_TRADING_API_KEY", "")
PAPER_TRADING_API_SECRET = os.getenv("PAPER_TRADING_API_SECRET", "")
PAPER_TRADING_ENDPOINT = os.getenv("PAPER_TRADING_ENDPOINT", "")

class AIDecisionAgent(BaseAgent):
    """AI-powered agent that makes trading decisions based on technical analysis and historical data"""
    
    def __init__(self, agent_id, risk_tolerance=0.5):
        super().__init__(agent_id, "AI_Decision")
        self.risk_tolerance = risk_tolerance
        self.model = None
        self.scaler = StandardScaler()
        self.historical_data = {}
        self.predictions = {}
        self.should_stop = False
        
    async def initialize(self):
        """Initialize the AI model"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.logger.info(f"AI Decision agent {self.agent_id} initialized with Random Forest model")
    
    async def run(self):
        """Main agent loop"""
        self.logger.info(f"AI Decision agent {self.agent_id} started")
        await self.initialize()
        
        try:
            while not self.should_stop:
                # Process messages from queue
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self.process_message(message)
                
                # Sleep to avoid high CPU usage
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in AI Decision agent: {str(e)}")
        finally:
            self.logger.info(f"AI Decision agent {self.agent_id} stopped")
        
    async def stop(self):
        """Stop the agent"""
        self.should_stop = True
        await asyncio.sleep(0.5)  # Give time for the run loop to exit
        
    async def train_model(self, symbol):
        """Train the AI model with historical data"""
        if symbol not in self.historical_data or len(self.historical_data[symbol]) < 30:
            self.logger.warning(f"Not enough historical data for {symbol} to train model")
            return False
            
        # Prepare training data
        data = self.historical_data[symbol]
        
        # Create features (technical indicators already calculated)
        features = []
        labels = []
        
        for i in range(20, len(data)):
            # Extract features from technical indicators
            feature_row = [
                data[i]['rsi'],
                data[i]['macd'],
                data[i]['macd_signal'],
                data[i]['macd_hist'],
                data[i]['bb_upper'],
                data[i]['bb_middle'],
                data[i]['bb_lower'],
                data[i]['volume'],
                data[i]['price'] / data[i-1]['price'] - 1,  # 1-day return
                data[i]['price'] / data[i-5]['price'] - 1,  # 5-day return
                data[i]['price'] / data[i-20]['price'] - 1,  # 20-day return
            ]
            features.append(feature_row)
            
            # Create label (1 for price increase after 5 days, 0 otherwise)
            if i + 5 < len(data):
                label = 1 if data[i+5]['price'] > data[i]['price'] else 0
            else:
                label = 1 if data[i]['price'] > data[i-1]['price'] else 0
            labels.append(label)
        
        if len(features) < 10:
            self.logger.warning(f"Not enough processed data points for {symbol} to train model")
            return False
            
        # Scale features
        X = self.scaler.fit_transform(features)
        y = np.array(labels)
        
        # Train model
        self.model.fit(X, y)
        self.logger.info(f"AI model trained for {symbol} with {len(X)} data points")
        return True
        
    async def predict(self, symbol, technical_data):
        """Make a prediction for the given symbol based on technical data"""
        if self.model is None:
            self.logger.warning("AI model not initialized")
            return None
            
        # Extract features from technical data
        try:
            feature_row = [
                technical_data.get('rsi', 50),
                technical_data.get('macd', 0),
                technical_data.get('macd_signal', 0),
                technical_data.get('macd_hist', 0),
                technical_data.get('bb_upper', 0),
                technical_data.get('bb_middle', 0),
                technical_data.get('bb_lower', 0),
                technical_data.get('volume', 0),
                technical_data.get('price_change_1d', 0),
                technical_data.get('price_change_5d', 0),
                technical_data.get('price_change_20d', 0),
            ]
            
            # Scale feature
            X = self.scaler.transform([feature_row])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]  # Probability of class 1 (price increase)
            
            self.predictions[symbol] = {
                'prediction': int(prediction),
                'probability': float(probability),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"AI prediction for {symbol}: {prediction} with probability {probability:.2f}")
            return {
                'prediction': int(prediction),
                'probability': float(probability)
            }
        except Exception as e:
            self.logger.error(f"Error making AI prediction: {str(e)}")
            return None
    
    async def process_message(self, message):
        """Process incoming messages"""
        content = message.get("content", {})
        msg_type = content.get("type", "")
        
        if msg_type == "technical_analysis_update":
            data = content.get("data", {})
            symbol = data.get("symbol", "")
            
            # Store historical data
            if symbol not in self.historical_data:
                self.historical_data[symbol] = []
            
            self.historical_data[symbol].append(data)
            
            # Limit historical data size
            if len(self.historical_data[symbol]) > 100:
                self.historical_data[symbol] = self.historical_data[symbol][-100:]
            
            # Train model if we have enough data
            if len(self.historical_data[symbol]) >= 30:
                await self.train_model(symbol)
            
            # Make prediction
            ai_prediction = await self.predict(symbol, data)
            
            # Combine AI prediction with technical analysis
            if ai_prediction:
                # Adjust the decision based on AI prediction and technical signals
                technical_signals = data.get("technical_signals", {})
                
                # Get the original recommendation
                original_recommendation = data.get("recommendation", "NEUTRAL")
                
                # Adjust based on AI prediction
                if ai_prediction['prediction'] == 1 and ai_prediction['probability'] > 0.6:
                    adjusted_recommendation = "BUY"
                    confidence_boost = ai_prediction['probability'] * 0.2
                elif ai_prediction['prediction'] == 0 and ai_prediction['probability'] > 0.6:
                    adjusted_recommendation = "SELL"
                    confidence_boost = ai_prediction['probability'] * 0.2
                else:
                    adjusted_recommendation = original_recommendation
                    confidence_boost = 0
                
                # Adjust confidence score
                original_confidence = data.get("confidence_score", 0.5)
                adjusted_confidence = min(0.95, original_confidence + confidence_boost)
                
                # Create AI-enhanced decision
                ai_decision = {
                    "symbol": symbol,
                    "recommendation": adjusted_recommendation,
                    "confidence_score": adjusted_confidence,
                    "suggested_position_size": adjusted_confidence * self.risk_tolerance,
                    "reasoning": {
                        "technical_signals": technical_signals,
                        "ai_prediction": {
                            "raw_prediction": ai_prediction['prediction'],
                            "probability": ai_prediction['probability'],
                            "confidence_boost": confidence_boost
                        }
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "is_ai_enhanced": True
                }
                
                # Send AI-enhanced decision
                await self.send_message({
                    "type": "ai_trading_decision",
                    "data": ai_decision
                })
                
                self.logger.info(f"Sent AI-enhanced trading decision for {symbol}: {adjusted_recommendation}")

class PaperTradingConnector:
    """Connector for paper trading platforms"""
    
    def __init__(self, platform="alpaca"):
        self.platform = platform
        self.api_key = PAPER_TRADING_API_KEY
        self.api_secret = PAPER_TRADING_API_SECRET
        self.endpoint = PAPER_TRADING_ENDPOINT
        self.client = None
        self.connected = False
        self.orders = []
        self.positions = {}
        
    async def connect(self):
        """Connect to the paper trading platform"""
        if not self.api_key or not self.api_secret:
            print(f"Paper trading credentials not configured. Check your .env file.")
            return False
            
        try:
            if self.platform == "alpaca":
                # Simulate connection to Alpaca
                print(f"Attempting to connect to Alpaca Paper Trading API at {self.endpoint}")
                print(f"Using API key: {self.api_key[:4]}...{self.api_key[-4:]}")
                
                try:
                    # Try to import alpaca-trade-api
                    import alpaca_trade_api as tradeapi
                    
                    # Create Alpaca client
                    self.client = tradeapi.REST(
                        self.api_key,
                        self.api_secret,
                        self.endpoint,
                        api_version='v2'
                    )
                    
                    # Test connection
                    try:
                        account = self.client.get_account()
                        print(f"Connected to Alpaca Paper Trading API")
                        print(f"Account ID: {account.id}")
                        print(f"Account Status: {account.status}")
                        self.connected = True
                        self.using_real_api = True
                    except Exception as e:
                        print(f"Error connecting to Alpaca API: {str(e)}")
                        print("Using simulated paper trading instead.")
                        self.connected = True
                        self.using_real_api = False
                except ImportError:
                    print("Alpaca Trade API not installed. Using simulated paper trading.")
                    print("Run 'pip install alpaca-trade-api' to install the required package.")
                    self.connected = True
                    self.using_real_api = False
            elif self.platform == "interactive_brokers":
                # Simulate connection to Interactive Brokers
                print(f"Connected to Interactive Brokers Paper Trading API")
                self.connected = True
                self.using_real_api = False
            else:
                print(f"Unsupported paper trading platform: {self.platform}")
                return False
                
            return self.connected
        except Exception as e:
            print(f"Error connecting to paper trading platform: {str(e)}")
            return False
    
    async def get_account_info(self):
        """Get account information"""
        if not self.connected:
            print("Not connected to paper trading platform")
            return None
            
        # Simulate account info
        return {
            "account_id": "PAPER123456",
            "buying_power": 100000.00,
            "cash": 100000.00,
            "portfolio_value": 100000.00,
            "platform": self.platform
        }
    
    async def place_order(self, symbol, quantity, side, order_type="market", limit_price=None):
        """Place an order"""
        if not self.connected:
            print("Not connected to paper trading platform")
            return None
            
        # Simulate order placement
        order_id = f"order_{len(self.orders) + 1}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "type": order_type,
            "limit_price": limit_price,
            "status": "filled",  # Simulate immediate fill
            "filled_quantity": quantity,
            "filled_price": limit_price if limit_price else random.uniform(100, 200),  # Simulate price
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.orders.append(order)
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {
                "symbol": symbol,
                "quantity": 0,
                "avg_price": 0,
                "market_value": 0
            }
            
        position = self.positions[symbol]
        
        if side == "buy":
            new_quantity = position["quantity"] + quantity
            new_avg_price = ((position["quantity"] * position["avg_price"]) + 
                            (quantity * order["filled_price"])) / new_quantity
            position["quantity"] = new_quantity
            position["avg_price"] = new_avg_price
        else:  # sell
            position["quantity"] -= quantity
            
        position["market_value"] = position["quantity"] * order["filled_price"]
        
        print(f"Order placed: {side.upper()} {quantity} {symbol} at {order['filled_price']:.2f}")
        return order
    
    async def get_positions(self):
        """Get current positions"""
        if not self.connected:
            print("Not connected to paper trading platform")
            return []
            
        return list(self.positions.values())
    
    async def get_orders(self, status=None):
        """Get orders"""
        if not self.connected:
            print("Not connected to paper trading platform")
            return []
            
        if status:
            return [order for order in self.orders if order["status"] == status]
        return self.orders

class StockAnalyzer:
    def __init__(self, symbol, exchange=None, use_ai=True, paper_trade=False):
        # Format symbol based on exchange
        if exchange:
            if exchange.upper() in ['NSE', 'BSE']:
                # For Indian stocks, format as NSE:TATAMOTORS or BSE:TATAMOTORS
                self.symbol = f"{exchange.upper()}:{symbol.upper()}"
                self.is_indian = True
            else:
                self.symbol = symbol.upper()
                self.is_indian = False
        else:
            # Check if symbol contains : which indicates exchange is already specified
            if ':' in symbol:
                self.symbol = symbol.upper()
                self.is_indian = symbol.split(':')[0].upper() in ['NSE', 'BSE']
            else:
                self.symbol = symbol.upper()
                self.is_indian = False
                
        self.result = None
        self.base_symbol = symbol.upper()
        self.use_ai = use_ai
        self.paper_trade = paper_trade
        self.paper_trading_connector = None
        
        if self.paper_trade:
            self.paper_trading_connector = PaperTradingConnector(PAPER_TRADING_API)
    
    async def analyze(self):
        """Analyze a single stock and return the recommendation"""
        print(f"\nAnalyzing {self.symbol}...")
        if self.is_indian:
            print(f"Processing as Indian stock on exchange: {self.symbol.split(':')[0]}")
        
        # Connect to paper trading if enabled
        if self.paper_trade:
            connected = await self.paper_trading_connector.connect()
            if connected:
                account_info = await self.paper_trading_connector.get_account_info()
                print(f"Connected to paper trading account: {account_info['account_id']}")
                print(f"Available buying power: ${account_info['buying_power']:.2f}")
        
        # Create agents
        comm_hub = CommunicationHubAgent("comm_hub_1")
        market_data = MarketDataAgent("market_data_1", [self.symbol])
        tech_analysis = TechnicalAnalysisAgent("tech_analysis_1")
        decision_maker = DecisionMakerAgent("decision_maker_1", 0.6)
        
        # Add AI decision agent if enabled
        ai_decision = None
        if self.use_ai:
            ai_decision = AIDecisionAgent("ai_decision_1", 0.7)
            print("AI-powered decision making enabled")
        
        # Set up subscriptions
        await comm_hub.subscribe("tech_analysis_1", ["market_data_update"])
        await comm_hub.subscribe("decision_maker_1", ["technical_analysis_update"])
        
        if self.use_ai:
            await comm_hub.subscribe("ai_decision_1", ["technical_analysis_update"])
        
        # Start agents
        agents = [comm_hub, market_data, tech_analysis, decision_maker]
        if self.use_ai:
            agents.append(ai_decision)
            
        agent_tasks = []
        
        for agent in agents:
            agent_tasks.append(asyncio.create_task(agent.start()))
            await asyncio.sleep(0.5)  # Small delay between starting agents
        
        try:
            # Wait for analysis to complete
            for _ in range(15):  # Wait up to 15 seconds
                await asyncio.sleep(1)
                
                # Check if we have a decision
                state = comm_hub.get_latest_state()
                trading_decisions = state.get("state", {}).get("trading_decisions", {})
                
                # Check for AI decision first if enabled
                if self.use_ai:
                    ai_decisions = state.get("state", {}).get("ai_trading_decisions", {})
                    if self.symbol in ai_decisions:
                        self.result = ai_decisions[self.symbol]
                        print(f"AI-enhanced decision received for {self.symbol}")
                        break
                
                # Fall back to regular decision
                if self.symbol in trading_decisions:
                    self.result = trading_decisions[self.symbol]
                    
                    # Check if using mock data
                    if "is_mock" in self.result:
                        print(f"Note: Using mock data for {self.symbol}. The symbol may not exist in Alpha Vantage or API limits reached.")
                    
                    break
            
            # If no result after waiting, check if we have any data
            if not self.result:
                print(f"No decision made for {self.symbol}. Checking if data was fetched...")
                
                # Try to get data directly from market data agent
                data = await market_data.fetch_market_data(self.symbol)
                if data:
                    if "is_mock" in data and data["is_mock"]:
                        print(f"Warning: Using mock data for {self.symbol}. The symbol may not exist or API limits reached.")
                        
                        # Force a decision using mock data
                        await tech_analysis.process_message({
                            "content": {
                                "type": "market_data_update",
                                "data": data
                            }
                        })
                        
                        # Wait a bit for the decision to be made
                        await asyncio.sleep(3)
                        
                        # Check again for a decision
                        state = comm_hub.get_latest_state()
                        trading_decisions = state.get("state", {}).get("trading_decisions", {})
                        
                        if self.symbol in trading_decisions:
                            self.result = trading_decisions[self.symbol]
                    else:
                        print(f"Data was fetched but no decision was made. Try again later.")
                else:
                    print(f"Error: Could not fetch data for {self.symbol}. The symbol may not exist.")
            
            # Execute paper trade if enabled and we have a decision
            if self.paper_trade and self.result and self.paper_trading_connector.connected:
                await self.execute_paper_trade()
        
        finally:
            # Stop all agents
            for agent in agents:
                await agent.stop()
            
            # Cancel all tasks
            for task in agent_tasks:
                task.cancel()
        
        return self.result
    
    async def execute_paper_trade(self):
        """Execute a paper trade based on the recommendation"""
        if not self.result:
            print("No recommendation available for paper trading")
            return
            
        recommendation = self.result.get("recommendation", "NEUTRAL")
        confidence = self.result.get("confidence_score", 0)
        position_size = self.result.get("suggested_position_size", 0)
        
        # Only trade if confidence is high enough
        if confidence < 0.3:
            print(f"Confidence too low ({confidence:.2f}) for paper trading")
            return
            
        # Get account info
        account_info = await self.paper_trading_connector.get_account_info()
        buying_power = account_info["buying_power"]
        
        # Calculate quantity based on position size and buying power
        # This is simplified - in reality you'd need to get the current price
        trade_value = buying_power * position_size
        price = 100  # Placeholder - would get actual price in real implementation
        quantity = int(trade_value / price)
        
        if quantity <= 0:
            print("Calculated quantity too small for trading")
            return
            
        # Place order based on recommendation
        if recommendation == "BUY":
            order = await self.paper_trading_connector.place_order(
                symbol=self.symbol,
                quantity=quantity,
                side="buy"
            )
            print(f"Paper trade executed: BUY {quantity} shares of {self.symbol}")
        elif recommendation == "SELL":
            order = await self.paper_trading_connector.place_order(
                symbol=self.symbol,
                quantity=quantity,
                side="sell"
            )
            print(f"Paper trade executed: SELL {quantity} shares of {self.symbol}")
        else:
            print(f"No paper trade executed for {self.symbol} - recommendation is NEUTRAL")
            
        # Update result with paper trade info
        if order:
            self.result["paper_trade"] = {
                "order_id": order["id"],
                "quantity": order["quantity"],
                "side": order["side"],
                "filled_price": order["filled_price"],
                "timestamp": order["timestamp"]
            }

def print_recommendation(result):
    """Print the recommendation in a nice format"""
    if not result:
        print("\nNo recommendation available.")
        return
    
    # Extract the base symbol without exchange prefix
    symbol = result.get('symbol', '')
    if ':' in symbol:
        display_symbol = symbol.split(':')[1]
        exchange = symbol.split(':')[0]
        display_title = f"{display_symbol} ({exchange})"
    else:
        display_title = symbol
    
    recommendation = result.get("recommendation", "UNKNOWN")
    confidence = result.get("confidence_score", 0)
    position_size = result.get("suggested_position_size", 0) * 100
    
    # Get technical signals
    signals = result.get("reasoning", {}).get("technical_signals", {})
    
    # Check if AI enhanced
    is_ai_enhanced = result.get("is_ai_enhanced", False)
    ai_prediction = result.get("reasoning", {}).get("ai_prediction", {})
    
    # Print header
    print("\n" + "=" * 60)
    print(f"TRADING RECOMMENDATION FOR {display_title}")
    if is_ai_enhanced:
        print("(AI-ENHANCED DECISION)")
    print("=" * 60)
    
    # Print recommendation
    if recommendation == "BUY":
        print(f"\n\033[92mRECOMMENDATION: BUY\033[0m")
    elif recommendation == "SELL":
        print(f"\n\033[91mRECOMMENDATION: SELL\033[0m")
    else:
        print(f"\n\033[93mRECOMMENDATION: NEUTRAL\033[0m")
    
    # Print details
    print(f"\nConfidence Score: {confidence:.2f}")
    print(f"Suggested Position Size: {position_size:.1f}%")
    
    # Print technical signals
    print("\nTECHNICAL SIGNALS:")
    print(f"MACD: {signals.get('macd', 'N/A')}")
    print(f"RSI: {signals.get('rsi', 'N/A')}")
    print(f"Bollinger Bands: {signals.get('bollinger', 'N/A')}")
    
    # Print AI prediction if available
    if is_ai_enhanced and ai_prediction:
        print("\nAI PREDICTION:")
        raw_prediction = ai_prediction.get("raw_prediction", -1)
        probability = ai_prediction.get("probability", 0)
        confidence_boost = ai_prediction.get("confidence_boost", 0)
        
        if raw_prediction == 1:
            print(f"Direction: \033[92mUPWARD\033[0m with {probability:.2f} probability")
        elif raw_prediction == 0:
            print(f"Direction: \033[91mDOWNWARD\033[0m with {probability:.2f} probability")
        
        print(f"AI Confidence Boost: +{confidence_boost:.2f}")
    
    # Print paper trade info if available
    paper_trade = result.get("paper_trade", None)
    if paper_trade:
        print("\nPAPER TRADE EXECUTED:")
        print(f"Order ID: {paper_trade.get('order_id', 'N/A')}")
        print(f"Quantity: {paper_trade.get('quantity', 0)}")
        print(f"Side: {paper_trade.get('side', 'N/A').upper()}")
        print(f"Filled Price: ${paper_trade.get('filled_price', 0):.2f}")
    
    # Print timestamp
    timestamp = datetime.fromisoformat(result.get("timestamp", datetime.utcnow().isoformat()))
    print(f"\nAnalysis Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 60)

async def main():
    # Check if symbol was provided
    if len(sys.argv) < 2:
        print("Please provide a stock symbol to analyze.")
        print("Usage: python analyze_stock.py SYMBOL [EXCHANGE] [OPTIONS]")
        print("Examples:")
        print("  python analyze_stock.py AAPL")
        print("  python analyze_stock.py TATAMOTORS NSE")
        print("  python analyze_stock.py RELIANCE BSE")
        print("\nOptions:")
        print("  --no-ai          Disable AI-enhanced decision making")
        print("  --paper-trade    Execute paper trades based on recommendations")
        return
    
    # Get the symbol from command line
    symbol = sys.argv[1]
    
    # Parse arguments
    args = sys.argv[2:]
    exchange = None
    use_ai = True
    paper_trade = False
    
    for arg in args:
        if arg.upper() in ['NSE', 'BSE']:
            exchange = arg
        elif arg == '--no-ai':
            use_ai = False
        elif arg == '--paper-trade':
            paper_trade = True
    
    # Create analyzer
    analyzer = StockAnalyzer(symbol, exchange, use_ai, paper_trade)
    
    # Analyze the stock
    result = await analyzer.analyze()
    
    # Print recommendation
    print_recommendation(result)

if __name__ == "__main__":
    asyncio.run(main()) 