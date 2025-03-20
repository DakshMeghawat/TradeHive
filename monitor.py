import asyncio
import threading
import os
import logging
from dotenv import load_dotenv
from main import TradingSystem
from report_generator import monitor_and_report

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TradingMonitor")

def run_report_monitor(trading_system, interval=300):
    """Run the report monitor in a separate thread"""
    monitor_thread = threading.Thread(
        target=monitor_and_report,
        args=(trading_system, interval),
        daemon=True
    )
    monitor_thread.start()
    return monitor_thread

async def main():
    # Get configuration from environment variables
    risk_tolerance = float(os.environ.get("RISK_TOLERANCE", 0.6))
    update_interval = int(os.environ.get("UPDATE_INTERVAL", 60))
    report_interval = int(os.environ.get("REPORT_INTERVAL", 300))  # 5 minutes by default
    
    # Configure the trading system
    symbols = ["AAPL", "MSFT"]
    
    logger.info(f"Starting trading system with: symbols={symbols}, risk_tolerance={risk_tolerance}")
    logger.info(f"Reports will be generated every {report_interval} seconds")
    
    # Create the trading system
    trading_system = TradingSystem(symbols, risk_tolerance, update_interval)
    
    # Start the report monitor in a separate thread
    monitor_thread = run_report_monitor(trading_system, report_interval)
    
    try:
        # Start the trading system
        await trading_system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Stop the trading system
        await trading_system.stop()
        logger.info("Trading system stopped")

if __name__ == "__main__":
    try:
        # Run the main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 