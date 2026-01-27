"""
Settings Screen.
"""
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.screen import Screen
from textual.widgets import Header, Footer, Button, Input, Switch, Label, Select, Static
from textual.message import Message

from src.app.config import AppConfig

class SettingsScreen(Screen):
    """Configuration Screen."""
    
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        
        with Container(classes="settings_root"):
            with Vertical(classes="box"):
                yield Label("Configuration", classes="title")
                
                with Grid(classes="settings_grid"):
                    # Column 1: General & Network
                    with Vertical(classes="settings_column"):
                        yield Label("TRADING", classes="section_header")
                        
                        with Horizontal(classes="row"):
                            yield Label("Mode:", classes="lbl")
                            yield Select.from_values(["paper", "real"], value=self.config.trading_mode, id="mode_select")

                        with Horizontal(classes="row"):
                            yield Label("Data Source:", classes="lbl")
                            yield Select.from_values(["polymarket", "onchain"], value=self.config.data_source, id="data_source_select")

                        with Horizontal(classes="row"):
                            yield Label("Model:", classes="lbl")
                            yield Input(value=self.config.model_path, id="model_path")
                            
                        yield Label("NETWORK", classes="section_header")
                        
                        with Horizontal(classes="row"):
                            yield Label("Proxy:", classes="lbl")
                            yield Input(value=self.config.proxy_url, id="proxy_url")
                            
                        with Horizontal(classes="row"):
                            yield Label("RPC:", classes="lbl")
                            yield Input(value=self.config.rpc_url, id="rpc_url")

                    # Column 2: Risk
                    with Vertical(classes="settings_column"):
                        yield Label("RISK MANAGEMENT", classes="section_header")
                        
                        with Horizontal(classes="row"):
                            yield Label("Max Loss/Day:", classes="lbl")
                            yield Input(value=str(self.config.risk.max_daily_loss_pct), id="max_daily_loss")

                        with Horizontal(classes="row"):
                            yield Label("Stop Loss (%):", classes="lbl")
                            yield Input(value=str(self.config.risk.stop_loss_pct), id="stop_loss")
                            
                        # Spacers or extra info could go here
                        
                with Horizontal(classes="actions"):
                    yield Button("Save Changes", variant="success", id="btn_save", classes="action_btn")
                    yield Button("Cancel", variant="error", id="btn_cancel", classes="action_btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save":
            self._save_settings()
            self.app.pop_screen()
            self.notify("Settings saved!")
        elif event.button.id == "btn_cancel":
            self.app.pop_screen()
            
    def _save_settings(self):
        # Update config object
        self.config.trading_mode = self.query_one("#mode_select", Select).value
        self.config.data_source = self.query_one("#data_source_select", Select).value
        self.config.model_path = self.query_one("#model_path", Input).value
        self.config.proxy_url = self.query_one("#proxy_url", Input).value
        self.config.rpc_url = self.query_one("#rpc_url", Input).value
        
        # Risk
        try:
            self.config.risk.max_daily_loss_pct = float(self.query_one("#max_daily_loss", Input).value)
            self.config.risk.stop_loss_pct = float(self.query_one("#stop_loss", Input).value)
        except ValueError:
            self.notify("Invalid number format in Risk settings", severity="error")
            return

        # Persist
        self.config.save()
