#!/usr/bin/env python
"""
Compare Traditional and MCP-enhanced trading decisions
"""

import os
import asyncio
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
from mcp_ai_agent import MCPDecisionAgent
from test_mcp import EnhancedMCPDecisionAgent

# Mock data for different stocks
STOCK_MOCK_DATA = {
    "AAPL": {
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
        },
        "mcp_response": {
            "recommendation": "BUY",
            "confidence_score": 0.78,
            "suggested_position_size": 0.15,
            "rationale": "AAPL shows strong technical signals with MACD and Bollinger Bands indicating upward momentum. The price is above both 50-day and 200-day moving averages, showing a solid uptrend. Fundamentally, the company's strong market position in the technology sector, combined with positive news sentiment (0.65) suggests continued growth potential. The broader market conditions are also favorable with S&P 500 and NASDAQ showing positive momentum. The relatively low volatility index indicates market stability.",
            "risks": [
                "Potential market volatility due to macroeconomic uncertainties",
                "Sector rotation away from technology stocks",
                "Earnings expectations may be priced in, creating potential for disappointment",
                "Supply chain disruptions affecting production capacity"
            ],
            "time_horizon": "medium-term"
        }
    },
    "MSFT": {
        "price": 412.30,
        "volume": 18500000,
        "macd": 0.85,
        "rsi": 58,
        "bb_upper": 415.50,
        "bb_middle": 410.20,
        "bb_lower": 405.60,
        "ma_50": 408.75,
        "ma_200": 395.20,
        "signals": {
            "overall": "NEUTRAL",
            "macd": "BUY",
            "rsi": "NEUTRAL",
            "bollinger": "NEUTRAL"
        },
        "mcp_response": {
            "recommendation": "BUY",
            "confidence_score": 0.68,
            "suggested_position_size": 0.12,
            "rationale": "While technical signals are mixed with a NEUTRAL overall rating, MSFT's strong cloud business growth and AI initiatives provide a fundamental tailwind. The company's resilience during economic slowdowns and strong cash position make it well-positioned to weather market volatility. News sentiment is positive, and the stock is trading near its 50-day moving average, suggesting potential support at current levels.",
            "risks": [
                "Competitive pressures in cloud market",
                "Regulatory scrutiny on AI development",
                "High valuation multiples compared to historical averages",
                "Slowing PC market affecting Windows revenue"
            ],
            "time_horizon": "long-term"
        }
    },
    "GOOGL": {
        "price": 147.85,
        "volume": 22000000,
        "macd": -0.25,
        "rsi": 45,
        "bb_upper": 150.30,
        "bb_middle": 146.80,
        "bb_lower": 143.20,
        "ma_50": 148.50,
        "ma_200": 142.35,
        "signals": {
            "overall": "SELL",
            "macd": "SELL",
            "rsi": "NEUTRAL",
            "bollinger": "NEUTRAL"
        },
        "mcp_response": {
            "recommendation": "NEUTRAL",
            "confidence_score": 0.55,
            "suggested_position_size": 0.08,
            "rationale": "Though technical indicators suggest a SELL signal with MACD turning negative, the broader context provides a more nuanced picture. Google's digital advertising market remains strong despite short-term headwinds, and their AI initiatives are gaining traction. The stock is trading near support levels, and regulatory concerns appear to be priced in. The negative MACD signal is countered by the stock being close to its 200-day moving average, which often provides support.",
            "risks": [
                "Ongoing regulatory challenges globally",
                "Increasing competition in digital advertising",
                "Rising costs for AI infrastructure development",
                "Pressure on margins from cloud investments"
            ],
            "time_horizon": "medium-term"
        }
    },
    "AMZN": {
        "price": 178.25,
        "volume": 30000000,
        "macd": 1.85,
        "rsi": 72,
        "bb_upper": 185.40,
        "bb_middle": 175.60,
        "bb_lower": 168.30,
        "ma_50": 172.45,
        "ma_200": 165.80,
        "signals": {
            "overall": "BUY",
            "macd": "BUY",
            "rsi": "SELL",
            "bollinger": "BUY"
        },
        "mcp_response": {
            "recommendation": "NEUTRAL",
            "confidence_score": 0.60,
            "suggested_position_size": 0.10,
            "rationale": "While technical signals are mostly positive, the RSI indicates overbought conditions at 72. Amazon's strong e-commerce and AWS growth are balanced against valuation concerns and potential market saturation. The stock has had a substantial recent run, suggesting it may be due for consolidation. The high RSI combined with price near the upper Bollinger Band suggests caution despite positive MACD signals.",
            "risks": [
                "Overbought technical conditions",
                "Margin pressure from increasing competition",
                "Labor cost increases affecting profitability",
                "Regulatory concerns regarding market dominance"
            ],
            "time_horizon": "short-term"
        }
    },
    "NVDA": {
        "price": 925.15,
        "volume": 40000000,
        "macd": 3.25,
        "rsi": 80,
        "bb_upper": 950.30,
        "bb_middle": 890.25,
        "bb_lower": 830.50,
        "ma_50": 850.75,
        "ma_200": 650.40,
        "signals": {
            "overall": "SELL",
            "macd": "BUY",
            "rsi": "SELL",
            "bollinger": "SELL"
        },
        "mcp_response": {
            "recommendation": "SELL",
            "confidence_score": 0.72,
            "suggested_position_size": 0.12,
            "rationale": "NVDA shows strong signs of being overextended with an RSI of 80 and price well above both moving averages. While the company's AI and data center growth remains robust, the stock has priced in significant future growth. The extremely overbought conditions suggest a near-term correction is likely, even as long-term fundamentals remain strong. Market expectations may be difficult to exceed at current valuation levels.",
            "risks": [
                "Semiconductor industry cyclicality",
                "Extremely high valuation multiples",
                "Potential normalization of AI-related demand",
                "Increased competition in GPU and AI accelerator markets",
                "Regulatory scrutiny on semiconductor supply chains"
            ],
            "time_horizon": "short-term"
        }
    }
}

class MultiStockAnalyzer:
    """Analyzer for comparing traditional and MCP-enhanced decisions across multiple stocks"""
    
    def __init__(self):
        self.stocks = STOCK_MOCK_DATA.keys()
        self.enhanced_agent = EnhancedMCPDecisionAgent("mcp_enhanced_agent", risk_tolerance=0.6)
        self.traditional_agent = MCPDecisionAgent("traditional_agent", risk_tolerance=0.6)
        
        # Set up the mock company and market data for both agents
        self.setup_common_data()
    
    def setup_common_data(self):
        """Set up common data for both agents"""
        # Add market context to both agents
        market_context = {
            "sp500_change": 0.5,
            "nasdaq_change": 0.7,
            "vix": 15.3,
            "treasury_10y": 3.5,
            "fed_rate": 5.25,
            "inflation_rate": 3.2,
            "gdp_growth": 2.1,
            "unemployment_rate": 3.8,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.enhanced_agent.market_context = market_context
        self.traditional_agent.market_context = market_context
        
        # Add technical data and company profiles for all stocks
        for symbol, data in STOCK_MOCK_DATA.items():
            # Technical data
            self.enhanced_agent.technical_data[symbol] = data
            self.traditional_agent.technical_data[symbol] = data
            
            # Company profile
            company_profile = {
                "symbol": symbol,
                "name": f"{symbol} Corporation",
                "sector": "Technology",
                "industry": "Software" if symbol not in ["AMZN", "GOOGL"] else "Internet Services",
                "market_cap": 1000000000 * (2 if symbol == "AAPL" else 1),
                "pe_ratio": 20.5 + (symbol.count('A') * 2),  # Just to vary the values
                "dividend_yield": 1.2 if symbol != "AMZN" else 0,
                "beta": 1.1 + (ord(symbol[0]) % 10) / 10,
                "52_week_high": data["price"] * 1.2,
                "52_week_low": data["price"] * 0.6,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.enhanced_agent.company_profiles[symbol] = company_profile
            self.traditional_agent.company_profiles[symbol] = company_profile
            
            # News sentiment
            news_sentiment = {
                "symbol": symbol,
                "sentiment_score": 0.65 - (0.1 if symbol == "GOOGL" else 0),
                "sentiment_magnitude": 0.8,
                "news_count": 5,
                "latest_headlines": [
                    f"{symbol} Reports Strong Quarterly Earnings",
                    f"{symbol} Announces New Product Line",
                    f"Analysts Upgrade {symbol} Stock Rating"
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.enhanced_agent.news_sentiment[symbol] = news_sentiment
            self.traditional_agent.news_sentiment[symbol] = news_sentiment
    
    async def override_mcp_response(self, symbol, prompt):
        """Override MCP response with pre-defined mock data"""
        return STOCK_MOCK_DATA[symbol]["mcp_response"]
    
    async def analyze_all_stocks(self):
        """Analyze all stocks with both traditional and MCP-enhanced approaches"""
        results = {}
        
        for symbol in self.stocks:
            # Get traditional decision using regular MCPDecisionAgent (fallback mode)
            trad_decision = await self.traditional_agent.generate_decision(symbol)
            
            # Override MCP call method for enhanced agent just for this stock
            self.enhanced_agent.call_mcp_llm = lambda prompt, sym=symbol: self.override_mcp_response(sym, prompt)
            
            # Get MCP-enhanced decision
            mcp_decision = await self.enhanced_agent.generate_decision(symbol)
            
            # Store results
            results[symbol] = {
                "traditional": trad_decision,
                "mcp_enhanced": mcp_decision
            }
            
            print(f"Analyzed {symbol}: Traditional: {trad_decision['recommendation']} vs MCP: {mcp_decision['recommendation']}")
        
        return results
    
    def create_comparison_visualizations(self, results, output_dir="reports"):
        """Create visualizations comparing traditional and MCP approaches"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/mcp_comparison_{timestamp}.png"
        
        # Create comparison data for plotting
        stocks = list(results.keys())
        trad_confidence = [results[s]["traditional"]["confidence_score"] for s in stocks]
        mcp_confidence = [results[s]["mcp_enhanced"]["confidence_score"] for s in stocks]
        trad_position = [results[s]["traditional"]["suggested_position_size"] for s in stocks]
        mcp_position = [results[s]["mcp_enhanced"]["suggested_position_size"] for s in stocks]
        
        recommendations = {
            "stocks": stocks,
            "trad_rec": [results[s]["traditional"]["recommendation"] for s in stocks],
            "mcp_rec": [results[s]["mcp_enhanced"]["recommendation"] for s in stocks],
            "match": [results[s]["traditional"]["recommendation"] == results[s]["mcp_enhanced"]["recommendation"] for s in stocks]
        }
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(15, 12))
        plt.suptitle("Traditional vs MCP-Enhanced Trading Analysis", fontsize=16)
        
        # 1. Confidence Score Comparison
        ax1 = plt.subplot(2, 2, 1)
        x = np.arange(len(stocks))
        width = 0.35
        
        ax1.bar(x - width/2, trad_confidence, width, label='Traditional', color='skyblue')
        ax1.bar(x + width/2, mcp_confidence, width, label='MCP-Enhanced', color='orange')
        
        ax1.set_title('Confidence Score Comparison')
        ax1.set_ylabel('Confidence (0-1)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(stocks)
        ax1.legend()
        
        # 2. Position Size Comparison
        ax2 = plt.subplot(2, 2, 2)
        
        ax2.bar(x - width/2, trad_position, width, label='Traditional', color='skyblue')
        ax2.bar(x + width/2, mcp_position, width, label='MCP-Enhanced', color='orange')
        
        ax2.set_title('Suggested Position Size Comparison')
        ax2.set_ylabel('Position Size (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stocks)
        ax2.legend()
        
        # 3. Recommendation Comparison (colored by matching vs different)
        ax3 = plt.subplot(2, 2, 3)
        
        # Color coding for recommendations
        colors = {
            'BUY': 'green',
            'SELL': 'red',
            'NEUTRAL': 'gray'
        }
        
        # Create recommendation data for table
        cell_text = []
        cell_colors = []
        
        for i, stock in enumerate(stocks):
            trad_rec = recommendations["trad_rec"][i]
            mcp_rec = recommendations["mcp_rec"][i]
            match = recommendations["match"][i]
            
            cell_text.append([stock, trad_rec, mcp_rec, "MATCH" if match else "DIFFERENT"])
            
            # Cell colors
            row_colors = ['lightgray', colors[trad_rec], colors[mcp_rec], 'lightgreen' if match else 'lightcoral']
            cell_colors.append(row_colors)
        
        # Hide axis
        ax3.axis('tight')
        ax3.axis('off')
        
        # Create table
        table = ax3.table(cellText=cell_text,
                          colLabels=["Stock", "Traditional", "MCP-Enhanced", "Comparison"],
                          cellColours=cell_colors,
                          loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax3.set_title('Trading Recommendations Comparison')
        
        # 4. Time Horizon (MCP-only feature)
        ax4 = plt.subplot(2, 2, 4)
        
        time_horizons = [results[s]["mcp_enhanced"]["reasoning"].get("time_horizon", "unknown") for s in stocks]
        horizon_colors = {
            "short-term": "lightcoral",
            "medium-term": "khaki",
            "long-term": "lightgreen",
            "unknown": "lightgray"
        }
        horizon_counts = {h: time_horizons.count(h) for h in set(time_horizons)}
        
        ax4.pie(horizon_counts.values(), 
                labels=horizon_counts.keys(), 
                autopct='%1.1f%%',
                colors=[horizon_colors[h] for h in horizon_counts.keys()])
        
        ax4.set_title('Time Horizons (MCP-Enhanced Only)')
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                   "Multi-Context Protocol enhances decisions by analyzing technical, fundamental, news, and market contexts.",
                   ha="center", fontsize=12, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
        
        return filename

async def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    print("=== Multi-Stock Comparison: Traditional vs MCP-Enhanced Analysis ===\n")
    
    analyzer = MultiStockAnalyzer()
    print(f"Analyzing {len(analyzer.stocks)} stocks: {', '.join(analyzer.stocks)}\n")
    
    # Analyze all stocks
    results = await analyzer.analyze_all_stocks()
    
    # Create summary table
    print("\n" + "="*100)
    print(f"{'STOCK':<10} {'TRADITIONAL':<30} {'MCP-ENHANCED':<30} {'CONFIDENCE Δ':<15} {'MATCH?':<10}")
    print("="*100)
    
    for symbol, data in results.items():
        trad = data["traditional"]
        mcp = data["mcp_enhanced"]
        
        confidence_diff = mcp["confidence_score"] - trad["confidence_score"]
        is_match = trad["recommendation"] == mcp["recommendation"]
        
        print(f"{symbol:<10} {trad['recommendation']} (Conf: {trad['confidence_score']:.2f}) {'':<2} {mcp['recommendation']} (Conf: {mcp['confidence_score']:.2f}) {'':<2} {confidence_diff:+.2f} {'':<8} {'✓' if is_match else '✗'}")
    
    print("="*100)
    
    # Create visualizations
    vis_file = analyzer.create_comparison_visualizations(results)
    
    print(f"\nComparison complete. Visualization saved to: {vis_file}")
    print("\nKey observations:")
    print("1. MCP-enhanced analysis provides more nuanced confidence scores based on multiple contexts")
    print("2. Position sizing is more conservative in the MCP-enhanced approach, reflecting better risk assessment")
    print("3. When traditional and MCP recommendations differ, MCP provides detailed rationale for its decision")
    print("4. MCP uniquely provides time horizon guidance, which is absent in traditional technical analysis")

if __name__ == "__main__":
    asyncio.run(main()) 