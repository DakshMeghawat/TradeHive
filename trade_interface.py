import os
import sys
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv
import time
import requests

# Import the StockAnalyzer class from analyze_stock.py
from analyze_stock import StockAnalyzer

# Load environment variables
load_dotenv()

# Paper trading API keys and settings
PAPER_TRADING_API_KEY = os.getenv("PAPER_TRADING_API_KEY", "")
PAPER_TRADING_API_SECRET = os.getenv("PAPER_TRADING_API_SECRET", "")
PAPER_TRADING_ENDPOINT = os.getenv("PAPER_TRADING_ENDPOINT", "")

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the header for the application"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "=" * 60)
    print("                   TRADEHIVE TRADING INTERFACE")
    print("=" * 60)
    print(f"Current Time: {current_time}")
    print("\nThis tool analyzes stocks and executes paper trades on Alpaca")
    print(f"API Key: {PAPER_TRADING_API_KEY[:4]}...{PAPER_TRADING_API_KEY[-4:]}")
    print(f"Endpoint: {PAPER_TRADING_ENDPOINT}")
    print("\n" + "-" * 60)

def print_menu():
    """Print the main menu"""
    print("\nMAIN MENU:")
    print("1. Analyze a stock")
    print("2. View current positions")
    print("3. View recent orders")
    print("4. View account information")
    print("5. Exit")
    
    return input("\nEnter your choice (1-5): ").strip()

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

def get_account_info():
    """Get account information from Alpaca"""
    headers = {
        "APCA-API-KEY-ID": PAPER_TRADING_API_KEY,
        "APCA-API-SECRET-KEY": PAPER_TRADING_API_SECRET
    }
    
    try:
        response = requests.get(f"{PAPER_TRADING_ENDPOINT}/v2/account", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting account information: {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to Alpaca: {str(e)}")
        return None

def get_positions():
    """Get current positions from Alpaca"""
    headers = {
        "APCA-API-KEY-ID": PAPER_TRADING_API_KEY,
        "APCA-API-SECRET-KEY": PAPER_TRADING_API_SECRET
    }
    
    try:
        response = requests.get(f"{PAPER_TRADING_ENDPOINT}/v2/positions", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting positions: {response.text}")
            return []
    except Exception as e:
        print(f"Error connecting to Alpaca: {str(e)}")
        return []

def get_orders():
    """Get recent orders from Alpaca"""
    headers = {
        "APCA-API-KEY-ID": PAPER_TRADING_API_KEY,
        "APCA-API-SECRET-KEY": PAPER_TRADING_API_SECRET
    }
    
    try:
        response = requests.get(f"{PAPER_TRADING_ENDPOINT}/v2/orders?status=all&limit=10", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting orders: {response.text}")
            return []
    except Exception as e:
        print(f"Error connecting to Alpaca: {str(e)}")
        return []

def display_account_info():
    """Display account information"""
    clear_screen()
    print_header()
    
    print("\nACCOUNT INFORMATION:")
    print("-" * 60)
    
    account = get_account_info()
    
    if account:
        print(f"Account ID: {account.get('id')}")
        print(f"Account Status: {account.get('status')}")
        print(f"Buying Power: ${float(account.get('buying_power', 0)):.2f}")
        print(f"Cash: ${float(account.get('cash', 0)):.2f}")
        print(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):.2f}")
        print(f"Equity: ${float(account.get('equity', 0)):.2f}")
        
        # Calculate day's change
        equity = float(account.get('equity', 0))
        last_equity = float(account.get('last_equity', 0))
        day_change = equity - last_equity
        day_change_percent = (day_change / last_equity) * 100 if last_equity > 0 else 0
        
        print(f"Day's Change: ${day_change:.2f} ({day_change_percent:.2f}%)")
    else:
        print("Failed to retrieve account information.")
    
    input("\nPress Enter to return to the main menu...")

def display_positions():
    """Display current positions"""
    clear_screen()
    print_header()
    
    print("\nCURRENT POSITIONS:")
    print("-" * 100)
    
    positions = get_positions()
    
    if positions:
        # Print header
        print(f"{'Symbol':<10} {'Quantity':<10} {'Avg Price':<15} {'Current Price':<15} {'Market Value':<15} {'P/L':<15} {'P/L %':<10}")
        print("-" * 100)
        
        # Print positions
        for position in positions:
            symbol = position.get('symbol', '')
            qty = position.get('qty', '0')
            avg_price = float(position.get('avg_entry_price', 0))
            current_price = float(position.get('current_price', 0))
            market_value = float(position.get('market_value', 0))
            unrealized_pl = float(position.get('unrealized_pl', 0))
            unrealized_plpc = float(position.get('unrealized_plpc', 0)) * 100
            
            print(f"{symbol:<10} {qty:<10} ${avg_price:<14.2f} ${current_price:<14.2f} ${market_value:<14.2f} ${unrealized_pl:<14.2f} {unrealized_plpc:<9.2f}%")
    else:
        print("No positions found.")
    
    input("\nPress Enter to return to the main menu...")

def display_orders():
    """Display recent orders"""
    clear_screen()
    print_header()
    
    print("\nRECENT ORDERS:")
    print("-" * 120)
    
    orders = get_orders()
    
    if orders:
        # Print header
        print(f"{'Symbol':<10} {'Side':<8} {'Quantity':<10} {'Type':<10} {'Status':<15} {'Created At':<25}")
        print("-" * 120)
        
        # Print orders
        for order in orders:
            symbol = order.get('symbol', '')
            side = order.get('side', '').upper()
            qty = order.get('qty', '0')
            order_type = order.get('type', '')
            status = order.get('status', '')
            
            # Format timestamp
            created_at = order.get('created_at', '')
            if created_at:
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            print(f"{symbol:<10} {side:<8} {qty:<10} {order_type:<10} {status:<15} {created_at:<25}")
    else:
        print("No orders found.")
    
    input("\nPress Enter to return to the main menu...")

def analyze_stock():
    """Analyze a stock and optionally execute a paper trade"""
    clear_screen()
    print_header()
    
    print("\nSTOCK ANALYZER:")
    print("-" * 60)
    
    # Get ticker from user
    ticker = input("\nEnter ticker symbol: ").strip().upper()
    
    if not ticker:
        print("Invalid ticker symbol.")
        time.sleep(2)
        return
    
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
    else:
        print("\nFailed to analyze ticker.")
    
    input("\nPress Enter to return to the main menu...")

def main():
    """Main function to run the trading interface"""
    while True:
        clear_screen()
        print_header()
        
        choice = print_menu()
        
        if choice == '1':
            analyze_stock()
        elif choice == '2':
            display_positions()
        elif choice == '3':
            display_orders()
        elif choice == '4':
            display_account_info()
        elif choice == '5':
            print("\nExiting. Thank you for using TradeHive Trading Interface!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main() 