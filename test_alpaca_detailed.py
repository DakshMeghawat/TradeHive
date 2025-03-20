import os
import requests
import json
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Get Alpaca API credentials
API_KEY = os.getenv("PAPER_TRADING_API_KEY")
API_SECRET = os.getenv("PAPER_TRADING_API_SECRET")
API_BASE_URL = os.getenv("PAPER_TRADING_ENDPOINT")

def test_alpaca_connection_direct():
    """Test connection to Alpaca paper trading account using direct API calls"""
    print("\n" + "=" * 60)
    print("TESTING ALPACA PAPER TRADING CONNECTION (DIRECT API CALLS)")
    print("=" * 60)
    
    # Print API credentials (masked for security)
    print(f"\nAPI Key: {API_KEY[:4]}...{API_KEY[-4:]}")
    print(f"API Secret: {API_SECRET[:4]}...{API_SECRET[-4:]}")
    print(f"API Base URL: {API_BASE_URL}")
    
    # Test account endpoint
    account_url = f"{API_BASE_URL}/v2/account"
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET
    }
    
    print(f"\nTesting account endpoint: {account_url}")
    print(f"Using headers: APCA-API-KEY-ID: {API_KEY[:4]}...{API_KEY[-4:]}")
    
    try:
        response = requests.get(account_url, headers=headers)
        print(f"\nResponse status code: {response.status_code}")
        
        if response.status_code == 200:
            account_data = response.json()
            print("\nConnection successful!")
            print("\nACCOUNT INFORMATION:")
            print(f"Account ID: {account_data.get('id')}")
            print(f"Account Status: {account_data.get('status')}")
            print(f"Buying Power: ${float(account_data.get('buying_power', 0)):.2f}")
            print(f"Cash: ${float(account_data.get('cash', 0)):.2f}")
            print(f"Portfolio Value: ${float(account_data.get('portfolio_value', 0)):.2f}")
            
            # Test positions endpoint
            positions_url = f"{API_BASE_URL}/v2/positions"
            print(f"\nTesting positions endpoint: {positions_url}")
            
            positions_response = requests.get(positions_url, headers=headers)
            print(f"Positions response status code: {positions_response.status_code}")
            
            if positions_response.status_code == 200:
                positions_data = positions_response.json()
                print(f"\nCurrent Positions: {len(positions_data)}")
                for position in positions_data:
                    print(f"  {position.get('symbol')}: {position.get('qty')} shares at ${float(position.get('avg_entry_price', 0)):.2f}, current value: ${float(position.get('market_value', 0)):.2f}")
            else:
                print(f"\nError getting positions: {positions_response.text}")
            
            # Test orders endpoint
            orders_url = f"{API_BASE_URL}/v2/orders?status=all&limit=10"
            print(f"\nTesting orders endpoint: {orders_url}")
            
            orders_response = requests.get(orders_url, headers=headers)
            print(f"Orders response status code: {orders_response.status_code}")
            
            if orders_response.status_code == 200:
                orders_data = orders_response.json()
                print(f"\nRecent Orders: {len(orders_data)}")
                for order in orders_data:
                    print(f"  {order.get('symbol')}: {order.get('side')} {order.get('qty')} shares, status: {order.get('status')}")
            else:
                print(f"\nError getting orders: {orders_response.text}")
            
            # Place a test order
            place_test_order = input("\nDo you want to place a test order for AAPL? (y/n): ")
            if place_test_order.lower() == 'y':
                order_url = f"{API_BASE_URL}/v2/orders"
                order_data = {
                    "symbol": "AAPL",
                    "qty": "1",
                    "side": "buy",
                    "type": "market",
                    "time_in_force": "day"
                }
                
                print(f"\nPlacing test order: {json.dumps(order_data)}")
                
                order_response = requests.post(order_url, headers=headers, json=order_data)
                print(f"Order response status code: {order_response.status_code}")
                
                if order_response.status_code in [200, 201]:
                    order_result = order_response.json()
                    print("\nTest order placed successfully!")
                    print(f"Order ID: {order_result.get('id')}")
                    print(f"Symbol: {order_result.get('symbol')}")
                    print(f"Side: {order_result.get('side')}")
                    print(f"Quantity: {order_result.get('qty')}")
                    print(f"Status: {order_result.get('status')}")
                    
                    # Wait a few seconds and check for the order
                    print("\nWaiting 5 seconds to check for the order...")
                    time.sleep(5)
                    
                    # Get the most recent order
                    check_order_url = f"{API_BASE_URL}/v2/orders/{order_result.get('id')}"
                    check_response = requests.get(check_order_url, headers=headers)
                    
                    if check_response.status_code == 200:
                        updated_order = check_response.json()
                        print(f"\nUpdated order status: {updated_order.get('status')}")
                    else:
                        print(f"\nError checking order: {check_response.text}")
                else:
                    print(f"\nError placing test order: {order_response.text}")
            
            return True
        else:
            print(f"\nError connecting to Alpaca: {response.text}")
            
            # Provide troubleshooting tips
            print("\nTROUBLESHOOTING TIPS:")
            print("1. Check that your API key and secret are correct")
            print("2. Make sure the API endpoint URL is correct (should be https://paper-api.alpaca.markets)")
            print("3. Verify that your Alpaca account is active")
            print("4. Check your internet connection")
            
            # Check if it's a 403 Forbidden error
            if response.status_code == 403:
                print("\nYou received a 403 Forbidden error. This usually means:")
                print("- Your API key or secret is incorrect")
                print("- Your API key doesn't have the necessary permissions")
                print("- Your account might be restricted")
                print("\nPlease verify your API credentials on the Alpaca dashboard.")
            
            return False
    except Exception as e:
        print(f"\nError making API request: {str(e)}")
        
        # Provide troubleshooting tips
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Check your internet connection")
        print("2. Verify the API endpoint URL is correct")
        print("3. Make sure you have the requests library installed (pip install requests)")
        
        return False

if __name__ == "__main__":
    test_alpaca_connection_direct() 