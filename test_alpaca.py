import os
import time
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables
load_dotenv()

# Get Alpaca API credentials
API_KEY = os.getenv("PAPER_TRADING_API_KEY")
API_SECRET = os.getenv("PAPER_TRADING_API_SECRET")
API_BASE_URL = os.getenv("PAPER_TRADING_ENDPOINT")

def test_alpaca_connection():
    """Test connection to Alpaca paper trading account"""
    print("\n" + "=" * 60)
    print("TESTING ALPACA PAPER TRADING CONNECTION")
    print("=" * 60)
    
    # Print API credentials (masked for security)
    print(f"\nAPI Key: {API_KEY[:4]}...{API_KEY[-4:]}")
    print(f"API Secret: {API_SECRET[:4]}...{API_SECRET[-4:]}")
    print(f"API Base URL: {API_BASE_URL}")
    
    try:
        # Initialize Alpaca API
        api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, api_version='v2')
        
        # Get account information
        account = api.get_account()
        print("\nConnection successful!")
        print("\nACCOUNT INFORMATION:")
        print(f"Account ID: {account.id}")
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${float(account.buying_power):.2f}")
        print(f"Cash: ${float(account.cash):.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
        
        # Get positions
        positions = api.list_positions()
        print(f"\nCurrent Positions: {len(positions)}")
        for position in positions:
            print(f"  {position.symbol}: {position.qty} shares at ${float(position.avg_entry_price):.2f}, current value: ${float(position.market_value):.2f}")
        
        # Get orders
        orders = api.list_orders(status='all', limit=10)
        print(f"\nRecent Orders: {len(orders)}")
        for order in orders:
            print(f"  {order.symbol}: {order.side} {order.qty} shares at ${float(order.limit_price) if order.limit_price else 0:.2f}, status: {order.status}")
        
        # Place a test order
        place_test_order = input("\nDo you want to place a test order for AAPL? (y/n): ")
        if place_test_order.lower() == 'y':
            try:
                # Submit a market order to buy 1 share of Apple at market price
                api.submit_order(
                    symbol='AAPL',
                    qty=1,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print("\nTest order placed successfully!")
                print("Please check your Alpaca dashboard to see the order.")
                print("Note: It may take a few seconds for the order to appear.")
                
                # Wait a few seconds and check for the order
                print("\nWaiting 5 seconds to check for the order...")
                time.sleep(5)
                
                # Get the most recent order
                orders = api.list_orders(status='all', limit=1)
                if orders:
                    order = orders[0]
                    print(f"\nMost recent order: {order.symbol} {order.side} {order.qty} shares, status: {order.status}")
                else:
                    print("\nNo recent orders found.")
            except Exception as e:
                print(f"\nError placing test order: {str(e)}")
        
        return True
    except Exception as e:
        print(f"\nError connecting to Alpaca: {str(e)}")
        
        # Provide troubleshooting tips
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Check that your API key and secret are correct")
        print("2. Make sure the API endpoint URL is correct (should be https://paper-api.alpaca.markets)")
        print("3. Verify that your Alpaca account is active")
        print("4. Check your internet connection")
        
        return False

if __name__ == "__main__":
    test_alpaca_connection() 