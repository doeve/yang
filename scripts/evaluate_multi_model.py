
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from pathlib import Path
from src.simulation.multi_asset_env import MultiAssetEnv, MultiAssetConfig
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Try importing RecurrentPPO
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

def evaluate_multi_model(
    model_path: str = "logs/recurrent_multi_asset_fixed/multi_asset_model",
    data_dir: str = "data",
    n_steps: int = 1000
):
    model_path = Path(model_path)
    if not model_path.exists():
        # Fallback to old path
        old_path = Path("logs/multi_asset/multi_asset_model")
        if old_path.exists():
            model_path = old_path
            
    print(f"Loading model from {model_path}...")
    
    # Load Model
    model = None
    is_recurrent = False
    
    if RecurrentPPO:
        try:
            model = RecurrentPPO.load(model_path)
            print("✓ Loaded RecurrentPPO (LSTM)")
            is_recurrent = True
        except:
            pass
            
    if model is None:
        try:
            model = SAC.load(model_path)
            print("✓ Loaded SAC (MLP)")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return

    print("Loading data...")
    print(f"  BTC: {Path(data_dir) / 'btcusdt_100ms.parquet'}")
    btc = pd.read_parquet(Path(data_dir) / "btcusdt_100ms.parquet")
    if "close" in btc.columns and "price" not in btc.columns:
        btc = btc.rename(columns={"close": "price"})

    dxy = None
    if (Path(data_dir) / 'dxy_1h.parquet').exists():
        dxy = pd.read_parquet(Path(data_dir) / "dxy_1h.parquet")

    eurusd = None
    if (Path(data_dir) / 'eurusd_1h.parquet').exists():
        eurusd = pd.read_parquet(Path(data_dir) / "eurusd_1h.parquet")

    print("Initializing environment...")
    config = MultiAssetConfig(random_start=True)
    
    # Wrap in VecEnv for normalization
    env = DummyVecEnv([lambda: MultiAssetEnv(btc, dxy, eurusd, config=config)])
    
    # Load normalization stats if available
    vec_path = model_path.parent / "vec_normalize.pkl"
    if vec_path.exists():
        print(f"✓ Loading normalization stats from {vec_path}")
        env = VecNormalize.load(str(vec_path), env)
        env.training = False # Don't update stats during eval
        env.norm_reward = False
    else:
        print("! No normalization stats found (obs may be unscaled)")

    print(f"\nEvaluating for {n_steps} steps...")
    
    correct = 0
    total_bets = 0
    up_bets = 0
    down_bets = 0
    position_sizes = []
    rewards = []
    
    # LSTM State
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    obs = env.reset()
    
    for _ in range(n_steps):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # VecEnv returns array of rewards/dones/infos
        reward = reward[0]
        done = done[0]
        inf = info[0]
        
        episode_starts[0] = done
        rewards.append(reward)
        
        if inf.get('position_taken'):
            total_bets += 1
            direction = inf['position_direction']
            size = inf['position_size']
            ret = inf['candle_return']
            
            position_sizes.append(size)
            
            if direction == 'UP':
                up_bets += 1
            else:
                down_bets += 1
                
            actual_dir = 'UP' if ret > 0 else 'DOWN'
            if direction == actual_dir:
                correct += 1
            
            # Print trades
            if total_bets <= 20:
                print(f"Trade {total_bets}: {direction} (Size: {size:.1%}) -> Actual: {actual_dir} ({ret:+.2%}) | Reward: {reward:.4f}")

    if total_bets > 0:
        accuracy = correct / total_bets
        avg_size = np.mean(position_sizes) if position_sizes else 0
        
        print("\n=== RESULTS ===")
        print(f"Accuracy:      {accuracy:.1%}")
        print(f"Total Bets:    {total_bets}")
        print(f"UP / DOWN:     {up_bets} / {down_bets}")
        print(f"Avg Size:      {avg_size:.1%}")
        print(f"Avg Reward:    {np.mean(rewards):.4f}")
    else:
        print("No bets taken.")

if __name__ == "__main__":
    evaluate_multi_model()
