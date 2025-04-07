# TradeHive

A comprehensive algorithmic trading system leveraging Multi-Context Protocol (MCP) for enhanced trading decisions.

## Features

- **MCP-Enhanced Analysis**: Combines technical analysis with fundamental data, news sentiment, and market context
- **Backtesting System**: Measure accuracy and performance of trading decisions
- **Trading Interface**: Interactive CLI for analyzing stocks and executing paper trades
- **Performance Comparison**: Visualizations comparing traditional vs MCP-enhanced trading

## Getting Started

### Prerequisites

```
python 3.8+
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with the following:

```
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER=True
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
MCP_API_KEY=your_mcp_api_key
MCP_API_URL=https://api.your-mcp-provider.com/v1/completions
```

### Usage

1. **Analyze a Stock**:
   ```
   python analyze_stock.py AAPL
   ```

2. **Run Backtest**:
   ```
   python backtest_mcp.py --symbols AAPL,MSFT,GOOGL
   ```

3. **Interactive Trading Interface**:
   ```
   python trade_interface.py
   ```

## Architecture

- **Decision Agents**: Both traditional and MCP-enhanced agents
- **Data Collection**: Technical, fundamental, news sentiment, and market context
- **Backtesting**: Forward simulation to measure accuracy
- **Trade Execution**: Paper trading through Alpaca API

## Performance

MCP-enhanced trading decisions have demonstrated significantly higher accuracy and returns compared to traditional technical analysis. The system uses visualizations to track and report performance metrics.
file for details.

**Disclaimer**: This system is for research and educational purposes only. Use at your own risk for actual trading. 
