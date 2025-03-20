import asyncio
import os
import sys
from dotenv import load_dotenv
import time

# Import the StockAnalyzer class from analyze_stock.py
from analyze_stock import StockAnalyzer

# Load environment variables
load_dotenv()

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the header for the application"""
    print("\n" + "=" * 60)
    print("TRADEHIVE STOCK ANALYZER")
    print("=" * 60)
    print("\nThis tool analyzes stocks and executes paper trades on Alpaca")
    print("API Key: " + os.getenv("PAPER_TRADING_API_KEY")[:4] + "..." + os.getenv("PAPER_TRADING_API_KEY")[-4:])
    print("Endpoint: " + os.getenv("PAPER_TRADING_ENDPOINT"))
    print("\n" + "-" * 60)

async def analyze_ticker(ticker, exchange=None, use_ai=True, paper_trade=True):
    """Analyze a ticker and execute a paper trade if requested"""
    try:
        # Create analyzer
        analyzer = StockAnalyzer(ticker, exchange, use_ai, paper_trade)
        
        # Analyze the stock
        result = await analyzer.analyze()
        
        return result
    except Exception as e:
        print(f"Error analyzing ticker: {str(e)}")
        return None

def main():
    """Main function to run the ticker analyzer"""
    clear_screen()
    print_header()
    
    while True:
        # Get ticker from user
        ticker = input("\nEnter ticker symbol (or 'q' to quit): ").strip().upper()
        
        if ticker.lower() == 'q':
            print("\nExiting. Thank you for using TradeHive Stock Analyzer!")
            break
        
        # Check if exchange is provided (for Indian stocks)
        exchange = None
        if ":" in ticker:
            parts = ticker.split(":")
            exchange = parts[0]
            ticker = parts[1]
        else:
            # Ask if it's an Indian stock
            is_indian = input("Is this an Indian stock? (y/n): ").strip().lower()
            if is_indian == 'y':
                exchange = input("Enter exchange (NSE/BSE): ").strip().upper()
        
        # Ask if paper trading should be enabled
        paper_trade = input("Execute paper trade? (y/n): ").strip().lower() == 'y'
        
        # Ask if AI should be used
        use_ai = input("Use AI-enhanced analysis? (y/n): ").strip().lower() == 'y'
        
        print(f"\nAnalyzing {ticker}" + (f" on {exchange}" if exchange else "") + "...")
        print("Please wait, this may take a few moments...\n")
        
        # Run the analysis
        result = asyncio.run(analyze_ticker(ticker, exchange, use_ai, paper_trade))
        
        if result:
            print("\nAnalysis complete!")
            
            # Ask if user wants to view another ticker
            another = input("\nAnalyze another ticker? (y/n): ").strip().lower()
            if another != 'y':
                print("\nExiting. Thank you for using TradeHive Stock Analyzer!")
                break
        else:
            print("\nFailed to analyze ticker. Please try again.")
            time.sleep(2)
        
        clear_screen()
        print_header()

if __name__ == "__main__":
    main() 