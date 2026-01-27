"""
Main Application Entry Point.
"""
import asyncio
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer

from src.app.config import AppConfig
from src.app.engine import TradingEngine
from src.app.dashboard import Dashboard

from src.app.settings import SettingsScreen

import logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


class YangApp(App):
    """Polymarket Trading TUI."""
    
    CSS_PATH = "yang.css"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
        ("s", "settings", "Settings"),
    ]
    
    def __init__(self):
        super().__init__()
        self.config = AppConfig.load()
        self.engine = TradingEngine(self.config)
        self.loop_task = None
        

    def on_mount(self):
        logging.info("App Mounting...")
        self.push_screen(Dashboard(id="dashboard"))
        # Start Engine in background
        self.run_worker(self.engine_lifespan())
        # Start UI update loop
        self.set_interval(1.0, self.update_ui)
        logging.info("App Mounted and Timers set")
        
    def action_settings(self):
        """Open settings screen."""
        self.push_screen(SettingsScreen(self.config))
        
    async def engine_lifespan(self):
        """Manage engine lifecycle."""
        await self.engine.start()
        while True:
            await self.engine.tick()
            await asyncio.sleep(1.0)
            

    async def update_ui(self):
        """Poll engine state and update widgets."""
        logging.info("UI Update Tick")

        try:
            # Access active screen directly
            dash = self.screen
            if not isinstance(dash, Dashboard):
                return # Settings or other screen active
                
            state = self.engine.get_state()
            market_data = state["market"]
            prediction = state["prediction"]
            dash.query_one("#left").update_data(market_data, self.config.trading_mode)

            # Account & Signal in RightPanel
            bal = await self.engine.execution.get_balance()
            slug = market_data.get("market_slug")
            pos = await self.engine.execution.get_position(slug) if slug else None
            
            # Get trades (not async in this implementation, or simply awaited)
            # engine.execution.get_trade_history is async
            trades = await self.engine.execution.get_trade_history()
            
            dash.query_one("#right").update_data(prediction, bal, pos, trades)
        except Exception as e:
            logging.error(f"UI Update Error: {e}", exc_info=True)


def main():
    app = YangApp()
    app.run()

if __name__ == "__main__":
    main()
