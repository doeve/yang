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
        self.push_screen(Dashboard(id="dashboard"))
        # Start Engine in background
        self.run_worker(self.engine_lifespan())
        # Start UI update loop
        self.set_interval(1.0, self.update_ui)
        
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
        try:
            dash = self.query_one(Dashboard)
        except:
            return # Dashboard not ready
            
        state = self.engine.get_state()
        market_data = state["market"]
        prediction = state["prediction"]
        
        dash.query_one("#market_watch").update_data(market_data)
        dash.query_one("#model_signal").update_data(prediction)
        
        # Account
        bal = await self.engine.execution.get_balance()
        slug = market_data.get("market_slug")
        pos = await self.engine.execution.get_position(slug) if slug else None
        
        dash.query_one("#account_summary").update_data(bal, pos)

def main():
    app = YangApp()
    app.run()

if __name__ == "__main__":
    main()
