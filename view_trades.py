import json
import os

def main():
    """Display paper trading information"""
    print("\n" + "=" * 60)
    print("TRADEHIVE PAPER TRADING DASHBOARD")
    print("=" * 60)
    
    # Check if files exist
    positions_file = "paper_trading_positions.json"
    orders_file = "paper_trading_orders.json"
    
    print("\nPOSITIONS:")
    if os.path.exists(positions_file):
        with open(positions_file, "r") as f:
            content = f.read()
            print(content)
    else:
        print(f"Error: {positions_file} not found")
    
    print("\nORDERS:")
    if os.path.exists(orders_file):
        with open(orders_file, "r") as f:
            content = f.read()
            print(content)
    else:
        print(f"Error: {orders_file} not found")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 