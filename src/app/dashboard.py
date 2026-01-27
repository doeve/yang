"""
Dashboard Screen and Widgets.
"""
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Label, Sparkline, DataTable
from textual.screen import Screen
import numpy as np

class MarketHeader(Static):
    """Row 1: General Info"""
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-container"):
            # Market Name
            with Vertical(classes="info-col"):
                yield Label("MARKET", classes="label-dim")
                yield Label("---", id="market_name", classes="value-bright")
            
            # BTC Price
            with Vertical(classes="info-col"):
                yield Label("BTC PRICE", classes="label-dim")
                with Horizontal(classes="centered"):
                     yield Label("---", id="btc_price", classes="value-bright")
                     yield Label("", id="btc_arrow", classes="arrow")
                
            # BTC Open
            with Vertical(classes="info-col"):
                yield Label("OPEN", classes="label-dim")
                yield Label("---", id="btc_open", classes="value-bright")

            # YES/NO Price
            with Vertical(classes="info-col"):
                yield Label("YES / NO", classes="label-dim")
                with Horizontal(classes="centered"):
                    yield Label("Y:", classes="label-dim")
                    yield Label("--", id="yes_price", classes="value-yes")
                    yield Label(" N:", classes="label-dim")
                    yield Label("--", id="no_price", classes="value-no")
                
            # Model & State
            with Vertical(classes="info-col"):
                yield Label("MODEL", classes="label-dim")
                yield Label("Unified v1", classes="value-bright")
                yield Label("---", id="app_state")

    def update_data(self, data: dict, config_mode: str = "paper"):
        # Market Name
        slug = data.get("market_slug", "---") or "---"
        self.query_one("#market_name", Label).update(slug.split("-")[-1] if "-" in slug else slug)
        
        # BTC
        btc = data.get("btc_current_price", 0)
        open_p = data.get("btc_open_price", 0)
        
        arrow = "⬆" if btc >= open_p else "⬇"
        color = "green" if btc >= open_p else "red"
        pct = ((btc - open_p) / open_p) * 100 if open_p else 0
        
        self.query_one("#btc_price", Label).update(f"[{color}]${btc:,.2f} ({pct:+.2f}%)[/]")
        self.query_one("#btc_arrow", Label).update(f"[{color}]{arrow}[/]")
        
        self.query_one("#btc_open", Label).update(f"${open_p:,.2f}")
        
        # YES/NO
        yes_p = data.get("yes_price", 0.5)
        no_p = data.get("no_price", 0.5)
        
        self.query_one("#yes_price", Label).update(f"{yes_p:.2f}")
        self.query_one("#no_price", Label).update(f"{no_p:.2f}")
        
        # State
        state_text = "[LIVE]" if config_mode == "real" else "[PAPER]"
        state_color = "red" if config_mode == "real" else "green"
        self.query_one("#app_state", Label).update(f"[{state_color}]{state_text}[/]")

class BTCChart(Static):
    """Row 2: BTC Chart"""
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("BTC TREND (vs Open)", classes="label-dim box-title")
            yield Sparkline([], summary_function=np.mean, id="btc_spark")
            
    def update_data(self, data: dict):
        hist = data.get("btc_price_history", [])
        if hist:
            self.query_one("#btc_spark", Sparkline).data = hist

class YESChart(Static):
    """Row 3: YES Chart"""
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("YES PRICE TREND", classes="label-dim box-title")
            yield Sparkline([], summary_function=np.mean, id="yes_spark")
            
    def update_data(self, data: dict):
        hist = data.get("yes_price_history", [])
        if hist:
            self.query_one("#yes_spark", Sparkline).data = hist

class LeftPanel(Static):
    """Left Side Panel (2/3 width)"""
    def compose(self) -> ComposeResult:
        with Vertical(id="left-container"):
            yield MarketHeader(id="row_header", classes="panel-row")
            yield BTCChart(id="row_btc", classes="panel-row")
            yield YESChart(id="row_yes", classes="panel-row")
        
    def update_data(self, data: dict, mode: str):
        self.query_one("#row_header").update_data(data, mode)
        self.query_one("#row_btc").update_data(data)
        self.query_one("#row_yes").update_data(data)

class RightPanel(Static):
    """Right Side Panel (1/3 width)"""
    def compose(self) -> ComposeResult:
        with Vertical(id="right-container"):
            # Signal
            with Vertical(classes="box-section"):
                yield Label("SIGNAL", classes="section-title")
                yield Label("---", id="signal_act", classes="big_text")
                yield Label("---", id="signal_conf", classes="small-text")
                
            # Position
            with Vertical(classes="box-section"):
                yield Label("POSITION", classes="section-title")
                yield Label("None", id="pos_info", classes="value-bright")
                yield Label("PnL: ---", id="pos_pnl")
                
            # Account
            with Vertical(classes="box-section"):
                yield Label("ACCOUNT", classes="section-title")
                yield Label("$---", id="balance", classes="value-bright")
                yield Label("Profit: ---", id="total_profit")
                
            # History
            with Vertical(classes="history-section"):
                yield Label("TRADE HISTORY", classes="section-title")
                yield DataTable(id="history_table", show_header=True, cursor_type="row")
            
    def on_mount(self):
        table = self.query_one("#history_table", DataTable)
        table.add_columns("Time", "Side", "Px", "PnL")
        
    def update_data(self, pred: dict, bal: float, pos, trades: list, initial_bal: float = 1000.0):
        # Signal
        act = pred.get("action", "WAIT")
        conf = pred.get("confidence", 0.0)
        
        if conf == 0 and act in ["WAIT", "HOLD"]:
            act = "BUFFERING"
            style = "grey"
        elif "BUY" in act: style = "green"
        elif "EXIT" in act: style = "orange"
        else: style = "grey"
        
        lbl = self.query_one("#signal_act", Label)
        lbl.update(act)
        lbl.classes = f"big_text bg-{style}"
        
        self.query_one("#signal_conf", Label).update(f"Conf: {conf:.1%}")
        
        # Position
        if pos:
            self.query_one("#pos_info", Label).update(f"{pos.side.upper()} {pos.size:.2f}")
            pnl_c = "green" if pos.pnl >= 0 else "red"
            self.query_one("#pos_pnl", Label).update(f"[{pnl_c}]${pos.pnl:+.2f}[/]")
        else:
            self.query_one("#pos_info", Label).update("None")
            self.query_one("#pos_pnl", Label).update("---")
            
        # Account
        self.query_one("#balance", Label).update(f"${bal:,.2f}")
        profit = bal - initial_bal
        prof_c = "green" if profit >= 0 else "red"
        self.query_one("#total_profit", Label).update(f"[{prof_c}]${profit:+.2f}[/]")
        
        # Trades
        table = self.query_one("#history_table", DataTable)
        
        # Check if we need to add rows
        # Simplistic sync: if len(trades) > rows, add difference
        # Assuming trades is append-only list
        current_rows = len(table.rows)
        if len(trades) > current_rows:
            for t in trades[current_rows:]:
                pnl_styled = f"[green]${t['pnl']:+.2f}[/]" if t['pnl'] >= 0 else f"[red]${t['pnl']:+.2f}[/]"
                table.add_row(
                    t["time"], 
                    t["side"].upper(), 
                    f"{t.get('exit', 0):.2f}",
                    pnl_styled
                )
            table.scroll_end(animate=False)

class Dashboard(Screen):
    CSS = """
    #main-layout {
        layout: horizontal;
        height: 1fr;
        width: 100%;
    }
    
    #left {
        width: 66%;
        height: 100%;
        border-right: solid green 50%;
    }
    
    #right {
        width: 34%;
        height: 100%;
    }
    
    #left-container {
        height: 100%;
    }

    .panel-row {
        height: 1fr;
        border-bottom: solid green 50%;
        padding: 1;
    }
    
    .header-container {
        height: 100%;
        align: center middle;
    }
    
    .info-col {
        width: 1fr;
        height: 100%;
        align: center middle;
    }
    
    .centered {
        align: center middle;
    }
    
    .label-dim { color: $text-disabled; text-align: center; }
    .box-title { margin-bottom: 1; }
    
    .value-bright { color: $text; text-style: bold; }
    .value-yes { color: green; text-style: bold; }
    .value-no { color: red; text-style: bold; }
    
    .box-section {
        height: auto;
        padding: 1;
        border-bottom: dashed grey;
    }
    
    .section-title {
        color: $text-disabled;
        margin-bottom: 1;
    }
    
    .big_text {
        text-align: center;
        text-style: bold;
        height: 3;
        content-align: center middle;
        background: $surface-lighten-1;
        color: $text;
        width: 100%;
    }
    
    .bg-green { background: green; color: black; }
    .bg-orange { background: orange; color: black; }
    .bg-grey { background: $surface-lighten-1; }
    
    .small-text { text-align: center; color: $text-muted; }
    
    .history-section {
        height: 1fr;
        padding: 0;
    }
    
    #btc_spark { color: blue; }
    #yes_spark { color: green; }
    
    DataTable {
        height: 100%;
        border: none;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            LeftPanel(id="left"),
            RightPanel(id="right"),
            id="main-layout"
        )
        yield Footer()
