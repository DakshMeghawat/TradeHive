import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def generate_visual_report(report_file):
    """Generate a visual report from a debug report file"""
    # Read the debug report
    with open(report_file, 'r') as f:
        content = f.read()
    
    # Extract the JSON data
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1
    json_str = content[start_idx:end_idx]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Error parsing JSON from {report_file}")
        return None
    
    # Check if we have trading decisions
    if 'state' not in data or 'trading_decisions' not in data['state'] or not data['state']['trading_decisions']:
        print("No trading decisions found in the report")
        return None
    
    # Extract trading decisions
    decisions = data['state']['trading_decisions']
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame([
        {
            'Symbol': symbol,
            'Recommendation': decision['recommendation'],
            'Confidence': decision['confidence_score'],
            'Position Size': decision['suggested_position_size'] * 100,  # Convert to percentage
            'MACD': decision['reasoning']['technical_signals']['macd'],
            'RSI': decision['reasoning']['technical_signals']['rsi'],
            'Bollinger': decision['reasoning']['technical_signals']['bollinger']
        }
        for symbol, decision in decisions.items()
    ])
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Recommendations
    colors = ['green' if rec == 'BUY' else 'red' if rec == 'SELL' else 'gray' for rec in df['Recommendation']]
    axes[0, 0].bar(df['Symbol'], df['Confidence'], color=colors)
    axes[0, 0].set_title('Trading Confidence by Symbol')
    axes[0, 0].set_ylabel('Confidence Score (0-1)')
    axes[0, 0].set_ylim(0, 1)
    
    # Add recommendation labels
    for i, rec in enumerate(df['Recommendation']):
        axes[0, 0].text(i, 0.05, rec, ha='center', va='bottom', color='white', fontweight='bold')
    
    # Plot 2: Position Sizes
    axes[0, 1].bar(df['Symbol'], df['Position Size'], color=colors)
    axes[0, 1].set_title('Suggested Position Size by Symbol')
    axes[0, 1].set_ylabel('Position Size (%)')
    axes[0, 1].set_ylim(0, max(df['Position Size']) * 1.2 if not df['Position Size'].empty else 10)
    
    # Plot 3: Technical Indicators
    indicator_counts = {
        'BUY': [sum(df[indicator] == 'BUY') for indicator in ['MACD', 'RSI', 'Bollinger']],
        'SELL': [sum(df[indicator] == 'SELL') for indicator in ['MACD', 'RSI', 'Bollinger']],
        'NEUTRAL': [sum(df[indicator] == 'NEUTRAL') for indicator in ['MACD', 'RSI', 'Bollinger']]
    }
    
    x = range(len(['MACD', 'RSI', 'Bollinger']))
    width = 0.25
    
    axes[1, 0].bar([i - width for i in x], indicator_counts['BUY'], width, label='BUY', color='green')
    axes[1, 0].bar([i for i in x], indicator_counts['SELL'], width, label='SELL', color='red')
    axes[1, 0].bar([i + width for i in x], indicator_counts['NEUTRAL'], width, label='NEUTRAL', color='gray')
    
    axes[1, 0].set_title('Technical Indicators Signals')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(['MACD', 'RSI', 'Bollinger'])
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # Plot 4: Summary
    summary = data['summary']
    summary_data = [summary['buy_signals'], summary['sell_signals'], summary['neutral_signals']]
    axes[1, 1].pie(summary_data, labels=['Buy', 'Sell', 'Neutral'], 
                  colors=['green', 'red', 'gray'], autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Signal Distribution')
    
    # Add timestamp
    plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ha='center', fontsize=10)
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reports/visual_report_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Visual report generated: {output_file}")
    return output_file

if __name__ == "__main__":
    # Find the latest debug report
    reports_dir = "reports"
    debug_reports = [f for f in os.listdir(reports_dir) if f.startswith("debug_report_")]
    
    if not debug_reports:
        print("No debug reports found")
    else:
        # Sort by timestamp (newest first)
        latest_report = sorted(debug_reports, reverse=True)[0]
        report_path = os.path.join(reports_dir, latest_report)
        
        print(f"Generating visual report from {report_path}")
        generate_visual_report(report_path) 