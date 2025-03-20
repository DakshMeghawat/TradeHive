from typing import Dict, Any, List
import asyncio
from datetime import datetime
from .base_agent import BaseAgent

class DecisionMakerAgent(BaseAgent):
    def __init__(self, agent_id: str, risk_tolerance: float = 0.5):
        super().__init__(agent_id, "DecisionMaker")
        self.risk_tolerance = risk_tolerance  # 0 to 1, where 1 is highest risk tolerance
        self.analysis_cache = {}
        self.pending_decisions = {}

    def _evaluate_opportunity(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trading opportunity based on technical and fundamental analysis"""
        technical_signals = data.get('signals', {})
        
        # Calculate confidence score (0-1)
        signal_weights = {
            'macd': 0.3,
            'rsi': 0.3,
            'bollinger': 0.4
        }
        
        confidence_score = 0
        for indicator, weight in signal_weights.items():
            if technical_signals.get(indicator) == technical_signals.get('overall'):
                confidence_score += weight

        # Determine position size based on confidence and risk tolerance
        base_position_size = 0.1  # 10% of available capital
        position_size = base_position_size * confidence_score * self.risk_tolerance

        return {
            'symbol': symbol,
            'recommendation': technical_signals.get('overall', 'NEUTRAL'),
            'confidence_score': confidence_score,
            'suggested_position_size': position_size,
            'reasoning': {
                'technical_signals': technical_signals,
                'risk_assessment': {
                    'risk_tolerance': self.risk_tolerance,
                    'position_size': position_size
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    async def process_message(self, message: Dict[str, Any]):
        """Process incoming messages from other agents"""
        content = message["content"]
        
        if content.get("type") == "technical_analysis_update":
            symbol = content["symbol"]
            analysis_data = content["data"]
            
            # Store the analysis
            self.analysis_cache[symbol] = {
                'technical_analysis': analysis_data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Generate trading decision
            decision = self._evaluate_opportunity(symbol, analysis_data)
            self.pending_decisions[symbol] = decision
            
            # Broadcast decision to communication hub
            await self.send_message(
                "comm_hub_1",
                {
                    "type": "trading_decision",
                    "decision": decision
                }
            )

    def _aggregate_decisions(self) -> List[Dict[str, Any]]:
        """Aggregate all pending decisions and prioritize opportunities"""
        decisions = list(self.pending_decisions.values())
        # Sort by confidence score and recommendation strength
        return sorted(
            decisions,
            key=lambda x: (x['confidence_score'], x['suggested_position_size']),
            reverse=True
        )

    async def run(self):
        """Main agent loop"""
        while self.running:
            try:
                # Process incoming messages
                while not self.message_queue.empty():
                    message = await self.receive_message()
                    await self.process_message(message)
                
                # Periodically aggregate and review decisions
                if self.pending_decisions:
                    prioritized_decisions = self._aggregate_decisions()
                    
                    # Send aggregated analysis to communication hub
                    await self.send_message(
                        "comm_hub_1",
                        {
                            "type": "aggregated_decisions",
                            "decisions": prioritized_decisions,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    
                    # Clear pending decisions after processing
                    self.pending_decisions.clear()
                
                await asyncio.sleep(5)  # Check for new decisions every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in decision maker agent main loop: {str(e)}")
                await asyncio.sleep(5) 