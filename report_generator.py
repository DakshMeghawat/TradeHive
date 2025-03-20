import json
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

class ReportGenerator:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.reports_dir = "reports"
        
        # Create reports directory if it doesn't exist
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def generate_report(self):
        """Generate a comprehensive report of the trading system state"""
        # Get current system state
        system_state = self.trading_system.get_system_state()
        
        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.reports_dir}/trading_report_{timestamp}.txt"
        
        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"TRADING SYSTEM REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # System Summary
            f.write("SYSTEM SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = system_state.get("summary", {})
            f.write(f"Active Symbols: {', '.join(summary.get('active_symbols', []))}\n")
            f.write(f"Total Decisions: {summary.get('total_decisions', 0)}\n")
            f.write(f"Buy Signals: {summary.get('buy_signals', 0)}\n")
            f.write(f"Sell Signals: {summary.get('sell_signals', 0)}\n")
            f.write(f"Neutral Signals: {summary.get('neutral_signals', 0)}\n\n")
            
            # Trading Decisions
            f.write("TRADING DECISIONS\n")
            f.write("-" * 80 + "\n")
            
            trading_decisions = system_state.get("state", {}).get("trading_decisions", {})
            if trading_decisions:
                decisions_data = []
                for symbol, decision in trading_decisions.items():
                    decisions_data.append([
                        symbol,
                        decision.get("recommendation", "UNKNOWN"),
                        f"{decision.get('confidence_score', 0):.2f}",
                        f"{decision.get('suggested_position_size', 0) * 100:.1f}%",
                        decision.get("timestamp", "")
                    ])
                
                if decisions_data:
                    f.write(tabulate(
                        decisions_data,
                        headers=["Symbol", "Recommendation", "Confidence", "Position Size", "Timestamp"],
                        tablefmt="grid"
                    ))
                    f.write("\n\n")
                else:
                    f.write("No trading decisions available yet.\n\n")
            else:
                f.write("No trading decisions available yet.\n\n")
            
            # Agent Status
            f.write("AGENT STATUS\n")
            f.write("-" * 80 + "\n")
            for agent_id, agent in self.trading_system.agents.items():
                f.write(f"{agent.agent_type} ({agent_id}): {'Running' if agent.running else 'Stopped'}\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Report generated: {report_file}")
        return report_file
    
    def generate_visual_report(self):
        """Generate visual charts for the trading system state"""
        # Get current system state
        system_state = self.trading_system.get_system_state()
        trading_decisions = system_state.get("state", {}).get("trading_decisions", {})
        
        if not trading_decisions:
            print("No trading decisions available for visualization yet.")
            return None
        
        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = f"{self.reports_dir}/trading_chart_{timestamp}.png"
        
        # Prepare data for visualization
        symbols = []
        recommendations = []
        confidence_scores = []
        position_sizes = []
        
        for symbol, decision in trading_decisions.items():
            symbols.append(symbol)
            rec = decision.get("recommendation", "NEUTRAL")
            recommendations.append(rec)
            confidence_scores.append(decision.get("confidence_score", 0))
            position_sizes.append(decision.get("suggested_position_size", 0) * 100)  # Convert to percentage
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot confidence scores
        bars = ax1.bar(symbols, confidence_scores, color=['green' if r == 'BUY' else 'red' if r == 'SELL' else 'gray' for r in recommendations])
        ax1.set_title('Trading Confidence by Symbol')
        ax1.set_ylabel('Confidence Score (0-1)')
        ax1.set_ylim(0, 1)
        
        # Add recommendation labels
        for bar, rec in zip(bars, recommendations):
            ax1.text(bar.get_x() + bar.get_width()/2, 0.05, rec, 
                    ha='center', va='bottom', color='white', fontweight='bold')
        
        # Plot position sizes
        ax2.bar(symbols, position_sizes, color=['green' if r == 'BUY' else 'red' if r == 'SELL' else 'gray' for r in recommendations])
        ax2.set_title('Suggested Position Size by Symbol')
        ax2.set_ylabel('Position Size (%)')
        ax2.set_ylim(0, max(position_sizes) * 1.2 if position_sizes else 10)
        
        plt.tight_layout()
        plt.savefig(chart_file)
        plt.close()
        
        print(f"Visual report generated: {chart_file}")
        return chart_file

def monitor_and_report(trading_system, interval=300):
    """Monitor the trading system and generate reports at regular intervals"""
    report_gen = ReportGenerator(trading_system)
    
    try:
        while True:
            # Generate text report
            report_file = report_gen.generate_report()
            
            # Generate visual report
            chart_file = report_gen.generate_visual_report()
            
            # Print the latest report to console
            if report_file:
                with open(report_file, 'r') as f:
                    print(f.read())
            
            # Wait for next report interval
            print(f"Next report in {interval} seconds. Press Ctrl+C to stop monitoring.")
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("Monitoring stopped.") 