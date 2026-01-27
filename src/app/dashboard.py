"""
Dashboard Screen and Widgets.
"""
from textual.app import ComposeResult
from textual.containers import Grid, Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Label, DataTable, Sparkline
from textual.screen import Screen
from rich.text import Text

class MarketWatch(Static):
    """Display current market details."""
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="box"):
            yield Label("Market Watch", classes="title")
            yield Label("Waiting for market...", id="mkt_slug")
            
            with Horizontal():
                yield Label("BTC: ", classes="label")
                yield Label("---", id="btc_price")
                
            with Horizontal():
                yield Label("Time Rem: ", classes="label")
                yield Label("---", id="time_rem")
                
            with Horizontal():
                yield Label("YES Price: ", classes="label")
                yield Label("---", id="yes_price")


    def update_data(self, data: dict):
        import logging
        logging.info(f"MarketWatch Update: {data.get('market_slug')}")
        self.query_one("#mkt_slug", Label).update(data.get("market_slug") or "Searching...")
        self.query_one("#btc_price", Label).update(f"${data.get('btc_current_price', 0):,.2f}")
        self.query_one("#time_rem", Label).update(f"{data.get('time_remaining', 0):.1%}")
        self.query_one("#yes_price", Label).update(f"{data.get('yes_price', 0):.3f}")


class ModelSignal(Static):
    """Display model prediction."""
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="box"):
            yield Label("Model Signal", classes="title")
            yield Label("---", id="signal_action", classes="big_text")
            
            with Horizontal():
                yield Label("Conf: ")
                yield Label("---", id="signal_conf")
                
            with Horizontal():
                yield Label("Exp Ret: ")
                yield Label("---", id="signal_ret")

    def update_data(self, pred: dict):
        if not pred:
            return
        
        act = pred.get("action", "WAIT")
        conf = pred.get("confidence", 0.0)
        ret = pred.get("expected_return", 0.0)
        
        lbl = self.query_one("#signal_action", Label)
        lbl.update(act)
        
        # Color code
        if "BUY" in act:
            lbl.classes = "big_text green"
        elif "EXIT" in act:
            lbl.classes = "big_text orange"
        else:
            lbl.classes = "big_text grey"
            
        self.query_one("#signal_conf", Label).update(f"{conf:.1%}")
        self.query_one("#signal_ret", Label).update(f"{ret:+.1%}")

class AccountSummary(Static):
    """Display account stats."""
    
    def compose(self) -> ComposeResult:
        with Vertical(classes="box"):
            yield Label("Account", classes="title")
            with Horizontal():
                yield Label("Balance: ")
                yield Label("$1000.00", id="balance")
            
            yield Label("Position", classes="subtitle")
            yield Label("None", id="pos_details")
            yield Label("PnL: $0.00", id="pos_pnl")

    def update_data(self, balance: float, position):
        self.query_one("#balance", Label).update(f"${balance:,.2f}")
        
        if position:
            self.query_one("#pos_details", Label).update(f"{position.side.upper()} x {position.size:.2f} @ {position.entry_price:.3f}")
            self.query_one("#pos_pnl", Label).update(f"PnL: ${position.pnl:+.2f}")
        else:
            self.query_one("#pos_details", Label).update("None")
            self.query_one("#pos_pnl", Label).update("---")

class Dashboard(Screen):
    """Main Trading Dashboard."""
    
    CSS = """
    .box {
        border: solid green;
        padding: 1;
        margin: 1;
        height: 1fr;
    }
    .title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $text;
        margin-bottom: 1;
    }
    .big_text {
        text-align: center;
        text-style: bold;
        height: 3;
        content-align: center middle;
    }
    .green { color: green; }
    .orange { color: orange; }
    .grey { color: grey; }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        
        with Grid(id="main_grid"):
            with Vertical():
                yield MarketWatch(id="market_watch")
                yield ModelSignal(id="model_signal")
            
            with Vertical():
                yield AccountSummary(id="account_summary")
                # Placeholder for charts or logs
                yield Static("Logs / History (Placeholder)", classes="box")
                
    def on_mount(self):
        # Set up grid layout
        pass
