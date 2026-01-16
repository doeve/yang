"""
FastAPI application for the ML trading system.

Provides REST API and WebSocket endpoints for:
- Training control and monitoring
- Simulation/backtesting
- Model management
- Real-time metrics streaming
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..data.storage import DataStorage
from ..data.collector import PolymarketCollector
from ..models.agent import TradingAgent, AgentConfig
from ..training.trainer import TrainingOrchestrator, TrainingConfig
from ..training.evaluation import Evaluator, generate_report
from ..simulation.environment import EnvConfig

logger = structlog.get_logger(__name__)


# =============================================================================
# Pydantic Models for API
# =============================================================================

class MarketInfo(BaseModel):
    """Market information response."""
    id: str
    question: str
    status: str
    has_price_data: bool
    price_data_points: int | None = None


class TrainingRequest(BaseModel):
    """Request to start training."""
    name: str = "training_run"
    total_timesteps: int = 100_000
    n_envs: int = 4
    market_ids: list[str] | None = None
    
    # Agent config
    extractor_type: str = "lstm"
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    features_dim: int = 64


class TrainingStatus(BaseModel):
    """Training run status."""
    id: int
    name: str
    status: str
    started_at: str | None
    completed_at: str | None
    total_episodes: int
    total_steps: int
    current_reward: float | None
    best_reward: float | None


class SimulationRequest(BaseModel):
    """Request to run simulation/backtest."""
    model_name: str
    market_id: str
    deterministic: bool = True


class SimulationResult(BaseModel):
    """Simulation result."""
    market_id: str
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    versions: list[dict[str, Any]]


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.storage: DataStorage | None = None
        self.active_training: dict[int, asyncio.Task] = {}
        self.websocket_clients: list[WebSocket] = []
        self.training_metrics: dict[int, list[dict]] = {}


state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    state.storage = DataStorage(data_dir)
    
    logger.info("Application started", data_dir=str(data_dir))
    
    yield
    
    # Shutdown
    for task in state.active_training.values():
        task.cancel()
    
    logger.info("Application shutdown")


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Polymarket ML Trader",
        description="Machine learning trading system for Polymarket prediction markets",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    _register_routes(app)
    
    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    
    # ==========================================================================
    # Health Check
    # ==========================================================================
    
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
    
    # ==========================================================================
    # Markets
    # ==========================================================================
    
    @app.get("/api/markets", response_model=list[MarketInfo])
    async def list_markets(
        has_price_data: bool | None = None,
        limit: int = 100,
    ):
        """List available markets."""
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage not initialized")
        
        markets = state.storage.get_markets(has_price_data=has_price_data, limit=limit)
        
        return [
            MarketInfo(
                id=m.id,
                question=m.question,
                status=m.status.value,
                has_price_data=True,  # Filter applied
                price_data_points=None,
            )
            for m in markets
        ]
    
    @app.post("/api/markets/collect")
    async def collect_market_data(background_tasks: BackgroundTasks):
        """Start collecting market data from Polymarket API."""
        
        async def _collect():
            async with PolymarketCollector() as collector:
                markets = await collector.fetch_crypto_markets()
                
                if state.storage:
                    state.storage.save_markets(markets)
                    
                    # Fetch price history for each market
                    for market in markets[:10]:  # Limit for demo
                        try:
                            for token in market.tokens:
                                token_id = token.get("token_id") or token.get("id")
                                if token_id:
                                    history = await collector.fetch_price_history(token_id)
                                    interpolated = await collector.interpolate_to_seconds(history)
                                    state.storage.save_price_history(interpolated)
                        except Exception as e:
                            logger.warning(f"Failed to fetch history", market_id=market.id, error=str(e))
        
        background_tasks.add_task(_collect)
        
        return {"status": "collection_started"}
    
    # ==========================================================================
    # Training
    # ==========================================================================
    
    @app.post("/api/training/start", response_model=TrainingStatus)
    async def start_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
    ):
        """Start a new training session."""
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage not initialized")
        
        # Create training config
        agent_config = AgentConfig(
            extractor_type=request.extractor_type,
            learning_rate=request.learning_rate,
            hidden_dim=request.hidden_dim,
            features_dim=request.features_dim,
        )
        
        training_config = TrainingConfig(
            total_timesteps=request.total_timesteps,
            n_envs=request.n_envs,
            agent_config=agent_config,
            experiment_name=request.name,
        )
        
        # Create training run
        run_id = state.storage.create_training_run(
            name=request.name,
            config={
                "total_timesteps": request.total_timesteps,
                "n_envs": request.n_envs,
                "extractor_type": request.extractor_type,
            },
        )
        
        # Start training in background
        async def _run_training():
            try:
                orchestrator = TrainingOrchestrator(state.storage, training_config)
                result = orchestrator.train(market_ids=request.market_ids)
                
                # Broadcast completion
                await _broadcast_message({
                    "type": "training_complete",
                    "run_id": run_id,
                    "result": result,
                })
            except Exception as e:
                logger.error("Training failed", run_id=run_id, error=str(e))
                if state.storage:
                    state.storage.update_training_run(run_id, status="failed")
        
        task = asyncio.create_task(_run_training())
        state.active_training[run_id] = task
        
        return TrainingStatus(
            id=run_id,
            name=request.name,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
            total_episodes=0,
            total_steps=0,
            current_reward=None,
            best_reward=None,
        )
    
    @app.get("/api/training/{run_id}/status", response_model=TrainingStatus)
    async def get_training_status(run_id: int):
        """Get status of a training run."""
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage not initialized")
        
        runs = state.storage.get_training_runs()
        run = next((r for r in runs if r["id"] == run_id), None)
        
        if not run:
            raise HTTPException(status_code=404, detail="Training run not found")
        
        return TrainingStatus(
            id=run["id"],
            name=run["name"],
            status=run["status"],
            started_at=run["started_at"],
            completed_at=run["completed_at"],
            total_episodes=run["total_episodes"],
            total_steps=run["total_steps"],
            current_reward=run.get("final_reward"),
            best_reward=run.get("best_reward"),
        )
    
    @app.post("/api/training/{run_id}/stop")
    async def stop_training(run_id: int):
        """Stop a running training session."""
        if run_id in state.active_training:
            state.active_training[run_id].cancel()
            del state.active_training[run_id]
            
            if state.storage:
                state.storage.update_training_run(run_id, status="stopped")
            
            return {"status": "stopped"}
        
        raise HTTPException(status_code=404, detail="Training run not found or not active")
    
    @app.get("/api/training", response_model=list[TrainingStatus])
    async def list_training_runs(limit: int = 20):
        """List recent training runs."""
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage not initialized")
        
        runs = state.storage.get_training_runs(limit=limit)
        
        return [
            TrainingStatus(
                id=r["id"],
                name=r["name"],
                status=r["status"],
                started_at=r["started_at"],
                completed_at=r["completed_at"],
                total_episodes=r["total_episodes"],
                total_steps=r["total_steps"],
                current_reward=r.get("final_reward"),
                best_reward=r.get("best_reward"),
            )
            for r in runs
        ]
    
    # ==========================================================================
    # Simulation
    # ==========================================================================
    
    @app.post("/api/simulation/run", response_model=SimulationResult)
    async def run_simulation(request: SimulationRequest):
        """Run a backtest simulation."""
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage not initialized")
        
        # Load model
        model_path = state.storage.get_model_path(request.model_name)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        agent = TradingAgent.load(model_path)
        
        # Load price data
        df = state.storage.load_price_data(request.market_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Market data not found")
        
        # Get market metadata
        market = state.storage.get_market(request.market_id)
        
        # Run backtest
        evaluator = Evaluator()
        result = evaluator.backtest(
            agent,
            df,
            market_id=request.market_id,
            resolution_at=market.resolution_at if market else None,
            created_at=market.created_at if market else None,
            outcome=market.outcome if market else None,
            deterministic=request.deterministic,
        )
        
        return SimulationResult(
            market_id=request.market_id,
            total_pnl=result.total_pnl,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
        )
    
    # ==========================================================================
    # Models
    # ==========================================================================
    
    @app.get("/api/models", response_model=list[ModelInfo])
    async def list_models():
        """List available trained models."""
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage not initialized")
        
        models = state.storage.list_models()
        
        return [
            ModelInfo(name=m["name"], versions=m["versions"])
            for m in models
        ]
    
    @app.get("/api/models/{name}/download")
    async def download_model(name: str, version: str = "latest"):
        """Download a trained model."""
        if state.storage is None:
            raise HTTPException(status_code=500, detail="Storage not initialized")
        
        model_path = state.storage.get_model_path(name, version)
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(
            model_path,
            filename=f"{name}_{version}.pt",
            media_type="application/octet-stream",
        )
    
    # ==========================================================================
    # WebSocket for Real-time Updates
    # ==========================================================================
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()
        state.websocket_clients.append(websocket)
        
        try:
            while True:
                # Keep connection alive and handle messages
                data = await websocket.receive_text()
                
                # Handle subscription commands
                if data.startswith("subscribe:"):
                    topic = data.split(":")[1]
                    await websocket.send_json({"status": "subscribed", "topic": topic})
                
        except WebSocketDisconnect:
            state.websocket_clients.remove(websocket)
    
    async def _broadcast_message(message: dict):
        """Broadcast message to all connected WebSocket clients."""
        for client in state.websocket_clients:
            try:
                await client.send_json(message)
            except Exception:
                state.websocket_clients.remove(client)


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
