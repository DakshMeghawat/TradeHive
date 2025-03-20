import asyncio
import os
from dotenv import load_dotenv
from main import TradingSystem
from report_generator import ReportGenerator

# Load environment variables
load_dotenv()

async def main():
    # Configure the trading system
    symbols = ["AAPL", "MSFT"]
    risk_tolerance = 0.6
    update_interval = 30  # Update every 30 seconds
    
    print(f"Starting trading system with symbols: {symbols}")
    print(f"Using Alpha Vantage API key: {os.environ.get('ALPHA_VANTAGE_API_KEY', 'Not set')}")
    
    # Create the trading system
    trading_system = TradingSystem(symbols, risk_tolerance, update_interval)
    
    # Initialize agents
    await trading_system.initialize_agents()
    
    # Start all agents
    agent_tasks = []
    
    # Start communication hub first
    print("Starting Communication Hub agent...")
    agent_tasks.append(asyncio.create_task(trading_system.agents["comm_hub"].start()))
    await asyncio.sleep(1)
    
    # Start market data agent
    print("Starting Market Data agent...")
    agent_tasks.append(asyncio.create_task(trading_system.agents["market_data"].start()))
    await asyncio.sleep(1)
    
    # Start technical analysis agent
    print("Starting Technical Analysis agent...")
    agent_tasks.append(asyncio.create_task(trading_system.agents["tech_analysis"].start()))
    await asyncio.sleep(1)
    
    # Start decision maker agent
    print("Starting Decision Maker agent...")
    agent_tasks.append(asyncio.create_task(trading_system.agents["decision_maker"].start()))
    
    # Create report generator
    report_gen = ReportGenerator(trading_system)
    
    try:
        # Run for a limited time
        for i in range(5):  # Generate 5 reports
            await asyncio.sleep(60)  # Wait 60 seconds between reports
            
            print("\n" + "="*80)
            print(f"GENERATING REPORT #{i+1}")
            print("="*80 + "\n")
            
            # Generate text report
            report_file = report_gen.generate_report()
            
            # Print the report content
            with open(report_file, 'r') as f:
                print(f.read())
        
        # Final visual report
        chart_file = report_gen.generate_visual_report()
        if chart_file:
            print(f"Visual report generated: {chart_file}")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        # Stop all agents
        for agent_name, agent in trading_system.agents.items():
            print(f"Stopping {agent_name} agent...")
            await agent.stop()
        
        # Cancel all tasks
        for task in agent_tasks:
            task.cancel()

if __name__ == "__main__":
    asyncio.run(main()) 