from typing import Dict, Any, List
import asyncio
from datetime import datetime
import json
from .base_agent import BaseAgent

class CommunicationHubAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "CommunicationHub")
        self.subscriptions = {}
        self.message_history = []
        self.latest_state = {}

    async def subscribe(self, agent_id: str, message_types: List[str]):
        """Subscribe an agent to specific message types"""
        for message_type in message_types:
            if message_type not in self.subscriptions:
                self.subscriptions[message_type] = set()
            self.subscriptions[message_type].add(agent_id)

    async def broadcast_message(self, message_type: str, content: Dict[str, Any]):
        """Broadcast message to all subscribed agents"""
        if message_type in self.subscriptions:
            for agent_id in self.subscriptions[message_type]:
                await self.send_message(
                    agent_id,
                    {
                        "type": message_type,
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )

    def _update_state(self, message: Dict[str, Any]):
        """Update the latest state based on incoming messages"""
        content = message["content"]
        message_type = content.get("type")

        if message_type == "trading_decision":
            decision = content.get("decision", {})
            symbol = decision.get("symbol")
            if symbol:
                if "trading_decisions" not in self.latest_state:
                    self.latest_state["trading_decisions"] = {}
                self.latest_state["trading_decisions"][symbol] = decision

        elif message_type == "aggregated_decisions":
            self.latest_state["aggregated_decisions"] = content.get("decisions", [])

        # Store message in history
        self.message_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message
        })

        # Trim history if it gets too long
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-1000:]

    def get_latest_state(self) -> Dict[str, Any]:
        """Get the latest state of the trading system"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "state": self.latest_state,
            "summary": self._generate_state_summary()
        }

    def _generate_state_summary(self) -> Dict[str, Any]:
        """Generate a summary of the current state"""
        summary = {
            "active_symbols": set(),
            "total_decisions": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "neutral_signals": 0
        }

        trading_decisions = self.latest_state.get("trading_decisions", {})
        for symbol, decision in trading_decisions.items():
            summary["active_symbols"].add(symbol)
            summary["total_decisions"] += 1
            
            recommendation = decision.get("recommendation", "NEUTRAL")
            if recommendation == "BUY":
                summary["buy_signals"] += 1
            elif recommendation == "SELL":
                summary["sell_signals"] += 1
            else:
                summary["neutral_signals"] += 1

        summary["active_symbols"] = list(summary["active_symbols"])
        return summary

    async def process_message(self, message: Dict[str, Any]):
        """Process incoming messages"""
        try:
            # Update internal state
            self._update_state(message)
            
            content = message["content"]
            message_type = content.get("type")

            # Broadcast message to subscribers
            await self.broadcast_message(message_type, content)

            # Log important state changes
            self.logger.info(f"Processed message type: {message_type}")
            if message_type in ["trading_decision", "aggregated_decisions"]:
                self.logger.info(f"Updated trading state: {json.dumps(self._generate_state_summary(), indent=2)}")

        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")

    async def run(self):
        """Main agent loop"""
        while self.running:
            try:
                # Process incoming messages
                while not self.message_queue.empty():
                    message = await self.receive_message()
                    await self.process_message(message)

                # Periodically clean up old messages
                current_time = datetime.utcnow()
                self.message_history = [
                    msg for msg in self.message_history
                    if (current_time - datetime.fromisoformat(msg["timestamp"])).days < 1
                ]

                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Error in communication hub agent main loop: {str(e)}")
                await asyncio.sleep(5) 