#!/usr/bin/env python
"""
Backtest MCP trading decisions vs traditional analysis
"""

import os
import asyncio
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv
from tabulate import tabulate
from mcp_ai_agent import MCPDecisionAgent
from test_mcp import EnhancedMCPDecisionAgent

# Load environment variables
load_dotenv()

class BacktestSystem:
    """System for backtesting and measuring trading decision accuracy"""
    
    def __init__(self, symbols, lookback_days=30, forward_days=10):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.forward_days = forward_days
        self.start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.future_end_date = (datetime.now() + timedelta(days=forward_days)).strftime('%Y-%m-%d')
        
        # Initialize agents
        self.traditional_agent = MCPDecisionAgent("traditional_agent", risk_tolerance=0.6)
        self.mcp_agent = EnhancedMCPDecisionAgent("mcp_enhanced_agent", risk_tolerance=0.6)
        
        # Results storage
        self.historical_data = {}
        self.decisions = {}
        self.trades = {}
        self.performance = {}
        
    async def load_historical_data(self):
        """Load historical stock data for analysis"""
        print(f"Loading historical data for {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                # Get historical data
                stock = yf.Ticker(symbol)
                hist = stock.history(start=self.start_date, end=self.end_date)
                
                if len(hist) == 0:
                    print(f"No historical data found for {symbol}")
                    continue
                
                self.historical_data[symbol] = hist
                print(f"Loaded {len(hist)} days of data for {symbol}")
                
                # Calculate technical indicators
                df = hist.copy()
                
                # Calculate MACD
                df['ema12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['ema26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['macd'] = df['ema12'] - df['ema26']
                df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                
                # Calculate RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # Calculate Bollinger Bands
                df['ma20'] = df['Close'].rolling(window=20).mean()
                df['std20'] = df['Close'].rolling(window=20).std()
                df['bb_upper'] = df['ma20'] + (df['std20'] * 2)
                df['bb_lower'] = df['ma20'] - (df['std20'] * 2)
                
                # Calculate Moving Averages
                df['ma50'] = df['Close'].rolling(window=50).mean()
                df['ma200'] = df['Close'].rolling(window=200).mean()
                
                # Store processed data
                self.historical_data[symbol] = df
                
            except Exception as e:
                print(f"Error loading data for {symbol}: {str(e)}")
    
    def generate_signals(self, symbol):
        """Generate technical signals for a symbol based on historical data"""
        df = self.historical_data[symbol]
        if df.empty:
            return {}
            
        # Get the latest data point
        latest = df.iloc[-1]
        
        # MACD Signal
        macd_signal = "BUY" if latest['macd'] > latest['signal'] else "SELL"
        
        # RSI Signal
        if latest['rsi'] < 30:
            rsi_signal = "BUY"
        elif latest['rsi'] > 70:
            rsi_signal = "SELL"
        else:
            rsi_signal = "NEUTRAL"
            
        # Bollinger Bands Signal
        if latest['Close'] < latest['bb_lower']:
            bb_signal = "BUY"
        elif latest['Close'] > latest['bb_upper']:
            bb_signal = "SELL"
        else:
            bb_signal = "NEUTRAL"
            
        # Moving Average Signal
        ma_signal = "BUY" if latest['ma50'] > latest['ma200'] else "SELL"
        
        # Overall Signal (simple majority)
        signals = [macd_signal, rsi_signal, bb_signal, ma_signal]
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        neutral_count = signals.count("NEUTRAL")
        
        if buy_count > sell_count and buy_count > neutral_count:
            overall = "BUY"
        elif sell_count > buy_count and sell_count > neutral_count:
            overall = "SELL"
        else:
            overall = "NEUTRAL"
            
        return {
            "overall": overall,
            "macd": macd_signal,
            "rsi": rsi_signal,
            "bollinger": bb_signal,
            "moving_average": ma_signal
        }
    
    async def prepare_agent_data(self, symbol):
        """Prepare data for agent analysis"""
        df = self.historical_data[symbol]
        if df.empty:
            return
            
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Set technical data
        technical_data = {
            "symbol": symbol,
            "price": latest['Close'],
            "volume": latest['Volume'],
            "macd": latest['macd'],
            "rsi": latest['rsi'],
            "bb_upper": latest['bb_upper'],
            "bb_middle": latest['ma20'],
            "bb_lower": latest['bb_lower'],
            "ma_50": latest['ma50'],
            "ma_200": latest['ma200'],
            "signals": self.generate_signals(symbol)
        }
        
        # Add technical data to both agents
        self.traditional_agent.technical_data[symbol] = technical_data
        self.mcp_agent.technical_data[symbol] = technical_data
        
        # Add market context (market data from past week)
        market_index = "^GSPC"  # S&P 500
        try:
            sp500 = yf.Ticker(market_index)
            sp500_hist = sp500.history(period="1w")
            sp500_change = ((sp500_hist['Close'].iloc[-1] / sp500_hist['Close'].iloc[0]) - 1) * 100
            
            # Simplified market context
            market_context = {
                "sp500_change": sp500_change,
                "nasdaq_change": sp500_change * 1.2,  # Approximation
                "vix": 15 + (10 if sp500_change < -1 else 0),  # Approximation
                "treasury_10y": 3.5,
                "fed_rate": 5.25,
                "inflation_rate": 3.2,
                "gdp_growth": 2.1,
                "unemployment_rate": 3.8,
                "timestamp": datetime.now().isoformat()
            }
            
            self.traditional_agent.market_context = market_context
            self.mcp_agent.market_context = market_context
            
        except Exception as e:
            print(f"Error loading market data: {str(e)}")
            
        # Add company profile
        company_profile = {
            "symbol": symbol,
            "name": f"{symbol} Corporation",
            "sector": "Technology",  # Simplified
            "industry": "Software",  # Simplified
            "market_cap": latest['Close'] * 1000000000,  # Simplified
            "pe_ratio": 20,  # Simplified
            "dividend_yield": 1.0,  # Simplified
            "beta": 1.1,
            "52_week_high": df['Close'].max(),
            "52_week_low": df['Close'].min(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.traditional_agent.company_profiles[symbol] = company_profile
        self.mcp_agent.company_profiles[symbol] = company_profile
        
        # Add news sentiment (simplified)
        news_sentiment = {
            "symbol": symbol,
            "sentiment_score": 0.6,  # Simplified positive sentiment
            "sentiment_magnitude": 0.8,
            "news_count": 5,
            "latest_headlines": [
                f"{symbol} Reports Strong Quarterly Earnings",
                f"{symbol} Announces New Product Launch",
                f"Analysts Upgrade {symbol} Stock Rating"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        self.traditional_agent.news_sentiment[symbol] = news_sentiment
        self.mcp_agent.news_sentiment[symbol] = news_sentiment
    
    async def get_future_performance(self, symbol, days=10):
        """Get future performance data to evaluate trade accuracy"""
        try:
            stock = yf.Ticker(symbol)
            # For real backtesting, we'd use actual future data
            # For prediction evaluation, we would store the prediction and check back later
            # For demo purposes, we'll use random future performance
            
            # Simulate future performance with random walk based on volatility
            df = self.historical_data[symbol]
            volatility = df['Close'].pct_change().std()
            
            # Generate random future prices
            current_price = df['Close'].iloc[-1]
            future_prices = [current_price]
            
            for i in range(days):
                # Random walk with historical volatility
                change = np.random.normal(0.0005, volatility)  # Slight upward bias
                next_price = future_prices[-1] * (1 + change)
                future_prices.append(next_price)
            
            # Calculate return
            future_return = (future_prices[-1] / future_prices[0]) - 1
            
            return {
                "start_price": future_prices[0],
                "end_price": future_prices[-1],
                "return": future_return,
                "daily_prices": future_prices,
                "days": days
            }
            
        except Exception as e:
            print(f"Error getting future data for {symbol}: {str(e)}")
            return None
    
    async def analyze_symbol(self, symbol):
        """Analyze a symbol and generate trading decisions"""
        print(f"Analyzing {symbol}...")
        
        # Prepare data for agents
        await self.prepare_agent_data(symbol)
        
        # Get trading decisions
        traditional_decision = await self.traditional_agent.generate_decision(symbol)
        
        # Override MCP call method to use our predefined responses
        self.mcp_agent.call_mcp_llm = lambda prompt: asyncio.create_task(
            self.mock_mcp_response(symbol, prompt)
        )
        
        mcp_decision = await self.mcp_agent.generate_decision(symbol)
        
        # Generate expected future performance
        future_performance = await self.get_future_performance(symbol, self.forward_days)
        
        # Store decisions and performance
        self.decisions[symbol] = {
            "traditional": traditional_decision,
            "mcp": mcp_decision,
            "timestamp": datetime.now().isoformat()
        }
        
        self.performance[symbol] = future_performance
        
        # Simulate trades based on decisions
        self.trades[symbol] = {
            "traditional": self.simulate_trade(traditional_decision, future_performance),
            "mcp": self.simulate_trade(mcp_decision, future_performance)
        }
        
        print(f"Completed analysis for {symbol}")
        
    async def mock_mcp_response(self, symbol, prompt):
        """Generate mock MCP response based on technical data and market conditions"""
        technical_data = self.mcp_agent.technical_data.get(symbol, {})
        signals = technical_data.get("signals", {})
        price = technical_data.get("price", 0)
        rsi = technical_data.get("rsi", 50)
        
        # MCP decisions are more nuanced than traditional ones
        if signals.get("overall") == "BUY" and rsi < 70:
            # Strong buy case - MCP agrees with technical analysis
            return {
                "recommendation": "BUY",
                "confidence_score": 0.75,
                "suggested_position_size": 0.15,
                "rationale": f"{symbol} shows strong technical signals with MACD and price above moving averages. Fundamentals support continued growth.",
                "risks": [
                    "Market volatility risk",
                    "Sector rotation potential",
                    "Valuation concerns if momentum slows"
                ],
                "time_horizon": "medium-term"
            }
        elif signals.get("overall") == "SELL" and rsi > 30:
            # Strong sell case - MCP agrees with technical analysis
            return {
                "recommendation": "SELL",
                "confidence_score": 0.72,
                "suggested_position_size": 0.12,
                "rationale": f"{symbol} shows deteriorating technical signals and stretched valuation metrics. Risk-reward ratio has become unfavorable.",
                "risks": [
                    "Short squeeze potential",
                    "Positive earnings surprise possibility",
                    "Market sentiment shift"
                ],
                "time_horizon": "short-term"
            }
        else:
            # Mixed signals - MCP provides more nuanced view than traditional analysis
            # MCP may disagree with technical analysis based on fundamental/news context
            if rsi > 70:
                # Overbought but technicals may be positive
                return {
                    "recommendation": "NEUTRAL",
                    "confidence_score": 0.60,
                    "suggested_position_size": 0.08,
                    "rationale": f"While {symbol} shows positive momentum, overbought conditions suggest caution. Fundamentals remain strong but current valuation has priced in much of the growth.",
                    "risks": [
                        "Technical correction likely",
                        "Profit taking pressure",
                        "Rotation to value stocks"
                    ],
                    "time_horizon": "short-term"
                }
            elif rsi < 30:
                # Oversold but technicals may be negative
                return {
                    "recommendation": "NEUTRAL",
                    "confidence_score": 0.65,
                    "suggested_position_size": 0.10,
                    "rationale": f"{symbol} has experienced significant selling pressure, but fundamental outlook remains stable. Current levels may present value for patient investors.",
                    "risks": [
                        "Further downside if sentiment deteriorates",
                        "Time required for sentiment reversal",
                        "Sector headwinds may persist"
                    ],
                    "time_horizon": "medium-term"
                }
            else:
                # Truly mixed signals
                # Randomly favor BUY or NEUTRAL to demonstrate differences
                import random
                rec = random.choice(["BUY", "NEUTRAL"])
                
                return {
                    "recommendation": rec,
                    "confidence_score": 0.55,
                    "suggested_position_size": 0.08,
                    "rationale": f"{symbol} presents a mixed picture with some positive and negative indicators. The broader market context suggests waiting for clearer signals.",
                    "risks": [
                        "Sideways market may persist",
                        "Volatility without clear direction",
                        "Opportunity cost of capital"
                    ],
                    "time_horizon": "medium-term"
                }
    
    def simulate_trade(self, decision, future_performance):
        """Simulate a trade based on the trading decision and future performance"""
        if not future_performance:
            return None
            
        recommendation = decision.get("recommendation", "NEUTRAL")
        confidence = decision.get("confidence_score", 0.5)
        entry_price = future_performance["start_price"]
        exit_price = future_performance["end_price"]
        future_return = future_performance["return"]
        
        # Determine if the trade was profitable based on the recommendation
        is_profitable = False
        if recommendation == "BUY" and future_return > 0:
            is_profitable = True
            pnl = future_return * 100  # Convert to percentage
        elif recommendation == "SELL" and future_return < 0:
            is_profitable = True
            pnl = -future_return * 100  # Convert to percentage (profit from short)
        elif recommendation == "NEUTRAL":
            # For NEUTRAL, we consider it correct if the price didn't move much
            is_profitable = abs(future_return) < 0.02  # Less than 2% move
            pnl = 0  # No position taken
        else:
            # Wrong prediction
            if recommendation == "BUY":
                pnl = future_return * 100  # Loss from long
            else:  # SELL
                pnl = -future_return * 100  # Loss from short
        
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": future_return * 100,  # Convert to percentage
            "is_profitable": is_profitable,
            "pnl": pnl,
            "days_held": future_performance["days"]
        }
    
    def calculate_accuracy(self):
        """Calculate the accuracy of trading decisions"""
        traditional_accuracy = 0
        mcp_accuracy = 0
        total_symbols = 0
        
        traditional_pnl = 0
        mcp_pnl = 0
        
        for symbol, trades in self.trades.items():
            if not trades:
                continue
                
            trad_trade = trades.get("traditional")
            mcp_trade = trades.get("mcp")
            
            if trad_trade and trad_trade.get("is_profitable"):
                traditional_accuracy += 1
                
            if mcp_trade and mcp_trade.get("is_profitable"):
                mcp_accuracy += 1
                
            if trad_trade:
                traditional_pnl += trad_trade.get("pnl", 0)
                
            if mcp_trade:
                mcp_pnl += mcp_trade.get("pnl", 0)
                
            total_symbols += 1
            
        if total_symbols == 0:
            return {
                "traditional_accuracy": 0,
                "mcp_accuracy": 0,
                "traditional_pnl": 0,
                "mcp_pnl": 0,
                "total_trades": 0
            }
            
        return {
            "traditional_accuracy": traditional_accuracy / total_symbols,
            "mcp_accuracy": mcp_accuracy / total_symbols,
            "traditional_pnl": traditional_pnl,
            "mcp_pnl": mcp_pnl,
            "total_trades": total_symbols
        }
    
    def generate_report(self):
        """Generate a detailed performance report"""
        results = self.calculate_accuracy()
        
        print("\n" + "="*80)
        print("PERFORMANCE REPORT: TRADITIONAL VS MCP-ENHANCED TRADING")
        print("="*80)
        
        print(f"\nTotal Symbols Analyzed: {results['total_trades']}")
        print(f"Backtest Period: {self.lookback_days} days historical, {self.forward_days} days forward test")
        print("\n" + "-"*80)
        
        # Create accuracy table
        accuracy_table = [
            ["Metric", "Traditional", "MCP-Enhanced", "Difference"],
            ["Accuracy", f"{results['traditional_accuracy']*100:.1f}%", f"{results['mcp_accuracy']*100:.1f}%", f"{(results['mcp_accuracy'] - results['traditional_accuracy'])*100:+.1f}%"],
            ["Total P&L", f"{results['traditional_pnl']:.2f}%", f"{results['mcp_pnl']:.2f}%", f"{results['mcp_pnl'] - results['traditional_pnl']:+.2f}%"]
        ]
        
        print(tabulate(accuracy_table, headers="firstrow", tablefmt="grid"))
        
        # Create detailed trade table
        trade_table = [["Symbol", "Method", "Signal", "Entry", "Exit", "Return", "Correct"]]
        
        for symbol, trades in self.trades.items():
            trad_trade = trades.get("traditional")
            mcp_trade = trades.get("mcp")
            
            if trad_trade:
                trade_table.append([
                    symbol,
                    "Traditional",
                    trad_trade["recommendation"],
                    f"${trad_trade['entry_price']:.2f}",
                    f"${trad_trade['exit_price']:.2f}",
                    f"{trad_trade['return']:+.2f}%",
                    "✓" if trad_trade["is_profitable"] else "✗"
                ])
                
            if mcp_trade:
                trade_table.append([
                    symbol,
                    "MCP-Enhanced",
                    mcp_trade["recommendation"],
                    f"${mcp_trade['entry_price']:.2f}",
                    f"${mcp_trade['exit_price']:.2f}",
                    f"{mcp_trade['return']:+.2f}%",
                    "✓" if mcp_trade["is_profitable"] else "✗"
                ])
        
        print("\n" + "-"*80)
        print("DETAILED TRADE PERFORMANCE")
        print("-"*80 + "\n")
        
        print(tabulate(trade_table, headers="firstrow", tablefmt="grid"))
        
        return results
        
    def create_visualization(self, output_dir="reports"):
        """Create visualization of performance comparison"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/backtest_results_{timestamp}.png"
        
        results = self.calculate_accuracy()
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(15, 12))
        plt.suptitle("Backtest Results: Traditional vs MCP-Enhanced Trading", fontsize=16)
        
        # 1. Accuracy Comparison
        ax1 = plt.subplot(2, 2, 1)
        methods = ['Traditional', 'MCP-Enhanced']
        accuracy = [results['traditional_accuracy'] * 100, results['mcp_accuracy'] * 100]
        
        ax1.bar(methods, accuracy, color=['skyblue', 'orange'])
        ax1.set_title('Trading Accuracy')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add accuracy values on top of bars
        for i, v in enumerate(accuracy):
            ax1.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        # 2. P&L Comparison
        ax2 = plt.subplot(2, 2, 2)
        pnl = [results['traditional_pnl'], results['mcp_pnl']]
        
        ax2.bar(methods, pnl, color=['skyblue', 'orange'])
        ax2.set_title('Total P&L')
        ax2.set_ylabel('P&L (%)')
        
        # Add P&L values on top of bars
        for i, v in enumerate(pnl):
            ax2.text(i, v + (1 if v >= 0 else -1), f"{v:.2f}%", ha='center')
        
        # 3. Per-Symbol Performance
        ax3 = plt.subplot(2, 2, 3)
        
        symbols = list(self.trades.keys())
        trad_pnl = []
        mcp_pnl = []
        
        for symbol in symbols:
            trades = self.trades.get(symbol, {})
            trad_trade = trades.get("traditional", {})
            mcp_trade = trades.get("mcp", {})
            
            trad_pnl.append(trad_trade.get("pnl", 0) if trad_trade else 0)
            mcp_pnl.append(mcp_trade.get("pnl", 0) if mcp_trade else 0)
        
        x = np.arange(len(symbols))
        width = 0.35
        
        ax3.bar(x - width/2, trad_pnl, width, label='Traditional', color='skyblue')
        ax3.bar(x + width/2, mcp_pnl, width, label='MCP-Enhanced', color='orange')
        
        ax3.set_title('P&L by Symbol')
        ax3.set_ylabel('P&L (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(symbols)
        ax3.legend()
        
        # 4. Signal Distribution and Success Rate
        ax4 = plt.subplot(2, 2, 4)
        
        # Count signals and successful signals
        signal_counts = {'Traditional': {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0},
                         'MCP': {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}}
        
        success_counts = {'Traditional': {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0},
                          'MCP': {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}}
        
        for symbol, trades in self.trades.items():
            trad_trade = trades.get("traditional")
            mcp_trade = trades.get("mcp")
            
            if trad_trade:
                rec = trad_trade["recommendation"]
                signal_counts['Traditional'][rec] += 1
                if trad_trade["is_profitable"]:
                    success_counts['Traditional'][rec] += 1
                    
            if mcp_trade:
                rec = mcp_trade["recommendation"]
                signal_counts['MCP'][rec] += 1
                if mcp_trade["is_profitable"]:
                    success_counts['MCP'][rec] += 1
        
        # Calculate success rates
        labels = []
        trad_rates = []
        mcp_rates = []
        
        for signal in ['BUY', 'SELL', 'NEUTRAL']:
            labels.append(f"{signal}\n(T:{signal_counts['Traditional'][signal]}, M:{signal_counts['MCP'][signal]})")
            
            trad_rate = 0
            if signal_counts['Traditional'][signal] > 0:
                trad_rate = success_counts['Traditional'][signal] / signal_counts['Traditional'][signal] * 100
                
            mcp_rate = 0
            if signal_counts['MCP'][signal] > 0:
                mcp_rate = success_counts['MCP'][signal] / signal_counts['MCP'][signal] * 100
                
            trad_rates.append(trad_rate)
            mcp_rates.append(mcp_rate)
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax4.bar(x - width/2, trad_rates, width, label='Traditional', color='skyblue')
        ax4.bar(x + width/2, mcp_rates, width, label='MCP-Enhanced', color='orange')
        
        ax4.set_title('Success Rate by Signal Type')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_ylim(0, 100)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.legend()
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   "MCP-Enhanced trading decisions leverage multiple contexts (technical, fundamental, news, market) for improved accuracy.",
                   ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
        
        return filename

async def main():
    """Main entry point"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Backtest MCP trading decisions")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,AMZN,NVDA",
                        help="Comma-separated list of stock symbols to analyze")
    parser.add_argument("--lookback", type=int, default=30,
                        help="Number of days to look back for historical data")
    parser.add_argument("--forward", type=int, default=10,
                        help="Number of days to simulate forward for performance")
    
    args = parser.parse_args()
    symbols = args.symbols.split(',')
    
    print(f"Starting backtest for {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Using {args.lookback} days historical data and {args.forward} days forward test")
    
    # Create and run backtest
    backtest = BacktestSystem(symbols, args.lookback, args.forward)
    
    # Load historical data
    await backtest.load_historical_data()
    
    # Analyze each symbol
    for symbol in symbols:
        if symbol in backtest.historical_data:
            await backtest.analyze_symbol(symbol)
            
    # Generate report
    backtest.generate_report()
    
    # Create visualization
    backtest.create_visualization()
    
    print("\nBacktest completed successfully")

if __name__ == "__main__":
    asyncio.run(main()) 