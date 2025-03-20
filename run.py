import os
import sys
import time
import asyncio
import threading
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the TradeHive header"""
    clear_screen()
    print("\033[92m")  # Green text
    print("=" * 80)
    print("""
████████╗██████╗  █████╗ ██████╗ ███████╗██╗  ██╗██╗██╗   ██╗███████╗
╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██║  ██║██║██║   ██║██╔════╝
   ██║   ██████╔╝███████║██║  ██║█████╗  ███████║██║██║   ██║█████╗  
   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██║██║╚██╗ ██╔╝██╔══╝  
   ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║██║ ╚████╔╝ ███████╗
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝  ╚══════╝
                                                                      
    """)
    print("=" * 80)
    print("\033[0m")  # Reset text color
    print("Multi-Agent Trading System")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

def print_menu():
    """Print the main menu"""
    print("\n\033[96mSelect an option:\033[0m")
    print("1. Run with real market data")
    print("2. Run with mock data (debug mode)")
    print("3. Run with monitoring and reporting")
    print("4. Generate visual report from latest data")
    print("5. View system configuration")
    print("6. Exit")
    print("\n")

def run_command(command):
    """Run a command and display its output"""
    clear_screen()
    print(f"\033[93mRunning: {command}\033[0m")
    print("-" * 80)
    os.system(command)
    print("\n" + "-" * 80)
    input("\nPress Enter to return to the menu...")

def view_configuration():
    """Display the current system configuration"""
    clear_screen()
    print("\033[93mSystem Configuration\033[0m")
    print("-" * 80)
    
    # API Key
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "Not set")
    if api_key != "Not set":
        masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    else:
        masked_key = "Not set (will use mock data)"
    
    print(f"Alpha Vantage API Key: {masked_key}")
    
    # Risk Tolerance
    risk_tolerance = os.environ.get("RISK_TOLERANCE", "0.6 (default)")
    print(f"Risk Tolerance: {risk_tolerance}")
    
    # Update Interval
    update_interval = os.environ.get("UPDATE_INTERVAL", "60 (default)")
    print(f"Update Interval: {update_interval} seconds")
    
    # Report Interval
    report_interval = os.environ.get("REPORT_INTERVAL", "300 (default)")
    print(f"Report Interval: {report_interval} seconds")
    
    # Symbols
    print(f"Monitored Symbols: AAPL, MSFT (default)")
    
    print("\n" + "-" * 80)
    input("\nPress Enter to return to the menu...")

def main():
    """Main function to run the TradeHive system"""
    while True:
        print_header()
        print_menu()
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == "1":
            run_command("python main.py")
        elif choice == "2":
            run_command("python debug_system.py")
        elif choice == "3":
            run_command("python monitor.py")
        elif choice == "4":
            run_command("python generate_report.py")
        elif choice == "5":
            view_configuration()
        elif choice == "6":
            clear_screen()
            print("\033[92mThank you for using TradeHive!\033[0m")
            print("Exiting...")
            sys.exit(0)
        else:
            print("\033[91mInvalid choice. Please try again.\033[0m")
            time.sleep(1)

if __name__ == "__main__":
    main() 