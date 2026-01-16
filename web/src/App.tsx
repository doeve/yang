import { useState, useEffect } from 'react';
import { Activity, Brain, BarChart3, Settings, PlayCircle, StopCircle, Download } from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Types
interface TrainingRun {
  id: number;
  name: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  total_episodes: number;
  total_steps: number;
  current_reward: number | null;
  best_reward: number | null;
}

interface Model {
  name: string;
  versions: { version: string; size_mb: number; modified_at: string }[];
}

interface SimulationResult {
  market_id: string;
  total_pnl: number;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
}

// API client
const api = {
  async get<T>(url: string): Promise<T> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  },
  
  async post<T>(url: string, data?: object): Promise<T> {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: data ? JSON.stringify(data) : undefined,
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  },
};

// Dashboard Stats Component
function StatsCard({ label, value, change, positive }: { 
  label: string; 
  value: string; 
  change?: string;
  positive?: boolean;
}) {
  return (
    <div className="card stagger-item">
      <div className="stat">
        <span className="stat-label">{label}</span>
        <span className={`stat-value ${positive === true ? 'positive' : positive === false ? 'negative' : ''}`}>
          {value}
        </span>
        {change && (
          <span className={`stat-change ${positive ? 'positive' : 'negative'}`}>
            {positive ? '↑' : '↓'} {change}
          </span>
        )}
      </div>
    </div>
  );
}

// Training Panel Component
function TrainingPanel({ onTrainingStart }: { onTrainingStart: () => void }) {
  const [config, setConfig] = useState({
    name: 'training_run',
    total_timesteps: 100000,
    n_envs: 4,
    extractor_type: 'lstm',
    learning_rate: 0.0003,
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await api.post('/api/training/start', config);
      onTrainingStart();
    } catch (err) {
      console.error('Failed to start training:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">
          <Brain size={20} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Training Configuration
        </h3>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label className="form-label">Run Name</label>
          <input
            type="text"
            className="form-input"
            value={config.name}
            onChange={(e) => setConfig({ ...config, name: e.target.value })}
          />
        </div>
        
        <div className="grid grid-cols-2" style={{ gap: '1rem' }}>
          <div className="form-group">
            <label className="form-label">Total Timesteps</label>
            <input
              type="number"
              className="form-input"
              value={config.total_timesteps}
              onChange={(e) => setConfig({ ...config, total_timesteps: parseInt(e.target.value) })}
            />
          </div>
          
          <div className="form-group">
            <label className="form-label">Parallel Environments</label>
            <input
              type="number"
              className="form-input"
              value={config.n_envs}
              onChange={(e) => setConfig({ ...config, n_envs: parseInt(e.target.value) })}
            />
          </div>
        </div>
        
        <div className="grid grid-cols-2" style={{ gap: '1rem' }}>
          <div className="form-group">
            <label className="form-label">Feature Extractor</label>
            <select
              className="form-input form-select"
              value={config.extractor_type}
              onChange={(e) => setConfig({ ...config, extractor_type: e.target.value })}
            >
              <option value="lstm">LSTM</option>
              <option value="transformer">Transformer</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </div>
          
          <div className="form-group">
            <label className="form-label">Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              className="form-input"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
            />
          </div>
        </div>
        
        <button 
          type="submit" 
          className="btn btn-primary btn-lg w-full mt-md"
          disabled={isLoading}
        >
          {isLoading ? (
            <span className="spinner" />
          ) : (
            <>
              <PlayCircle size={18} />
              Start Training
            </>
          )}
        </button>
      </form>
    </div>
  );
}

// Training Runs List Component
function TrainingRunsList({ runs, onRefresh }: { runs: TrainingRun[]; onRefresh: () => void }) {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'running':
        return <span className="badge badge-info">Running</span>;
      case 'completed':
        return <span className="badge badge-success">Completed</span>;
      case 'failed':
        return <span className="badge badge-error">Failed</span>;
      default:
        return <span className="badge">{status}</span>;
    }
  };

  const handleStop = async (runId: number) => {
    try {
      await api.post(`/api/training/${runId}/stop`);
      onRefresh();
    } catch (err) {
      console.error('Failed to stop training:', err);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">
          <Activity size={20} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Training Runs
        </h3>
      </div>
      
      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Status</th>
              <th>Steps</th>
              <th>Best Reward</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {runs.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', padding: '2rem' }}>
                  No training runs yet. Start one above!
                </td>
              </tr>
            ) : (
              runs.map((run) => (
                <tr key={run.id}>
                  <td>{run.name}</td>
                  <td>{getStatusBadge(run.status)}</td>
                  <td>{run.total_steps.toLocaleString()}</td>
                  <td>
                    {run.best_reward !== null ? (
                      <span className={run.best_reward > 0 ? 'positive' : 'negative'}>
                        {run.best_reward.toFixed(2)}
                      </span>
                    ) : '-'}
                  </td>
                  <td>
                    {run.status === 'running' && (
                      <button 
                        className="btn btn-sm btn-danger"
                        onClick={() => handleStop(run.id)}
                      >
                        <StopCircle size={14} />
                        Stop
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Learning Curve Chart Component
function LearningCurveChart({ data }: { data: { step: number; reward: number }[] }) {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">
          <BarChart3 size={20} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Learning Progress
        </h3>
      </div>
      
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis 
              dataKey="step" 
              stroke="#6b7280"
              tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
            />
            <YAxis stroke="#6b7280" />
            <Tooltip
              contentStyle={{
                background: '#1a2234',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '0.5rem',
              }}
            />
            <Area
              type="monotone"
              dataKey="reward"
              stroke="#6366f1"
              strokeWidth={2}
              fill="url(#rewardGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// Models List Component
function ModelsList({ models }: { models: Model[] }) {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">
          <Settings size={20} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Saved Models
        </h3>
      </div>
      
      {models.length === 0 ? (
        <p style={{ textAlign: 'center', padding: '2rem', color: 'var(--color-text-muted)' }}>
          No models saved yet
        </p>
      ) : (
        <div className="flex flex-col gap-md">
          {models.map((model) => (
            <div 
              key={model.name}
              className="flex items-center justify-between p-md"
              style={{ 
                background: 'var(--color-bg-tertiary)', 
                borderRadius: 'var(--radius-md)' 
              }}
            >
              <div>
                <div style={{ fontWeight: 600 }}>{model.name}</div>
                <div style={{ fontSize: '0.875rem', color: 'var(--color-text-muted)' }}>
                  {model.versions.length} version(s)
                </div>
              </div>
              <button className="btn btn-secondary btn-sm">
                <Download size={14} />
                Export
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Simulation Result Card
function SimulationResultCard({ result }: { result: SimulationResult }) {
  return (
    <div className="card">
      <div className="card-header">
        <h3 className="card-title">Simulation Results</h3>
        <span className="badge badge-success">Completed</span>
      </div>
      
      <div className="grid grid-cols-3" style={{ gap: '1rem' }}>
        <div className="stat">
          <span className="stat-label">Total PnL</span>
          <span className={`stat-value ${result.total_pnl >= 0 ? 'positive' : 'negative'}`}>
            ${result.total_pnl.toFixed(2)}
          </span>
        </div>
        
        <div className="stat">
          <span className="stat-label">Sharpe Ratio</span>
          <span className="stat-value">{result.sharpe_ratio.toFixed(2)}</span>
        </div>
        
        <div className="stat">
          <span className="stat-label">Max Drawdown</span>
          <span className="stat-value negative">
            {(result.max_drawdown * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="stat">
          <span className="stat-label">Win Rate</span>
          <span className="stat-value">{(result.win_rate * 100).toFixed(1)}%</span>
        </div>
        
        <div className="stat">
          <span className="stat-label">Total Trades</span>
          <span className="stat-value">{result.total_trades}</span>
        </div>
        
        <div className="stat">
          <span className="stat-label">Return</span>
          <span className={`stat-value ${result.total_return >= 0 ? 'positive' : 'negative'}`}>
            {(result.total_return * 100).toFixed(2)}%
          </span>
        </div>
      </div>
    </div>
  );
}

// Main App Component
function App() {
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [activeTab, setActiveTab] = useState<'training' | 'simulation' | 'models'>('training');
  
  // Mock learning curve data
  const [learningData] = useState(() => 
    Array.from({ length: 50 }, (_, i) => ({
      step: i * 2000,
      reward: Math.sin(i / 10) * 50 + 100 + Math.random() * 20 + i * 2,
    }))
  );

  // Fetch data
  const fetchRuns = async () => {
    try {
      const data = await api.get<TrainingRun[]>('/api/training');
      setRuns(data);
    } catch (err) {
      console.error('Failed to fetch runs:', err);
    }
  };

  const fetchModels = async () => {
    try {
      const data = await api.get<Model[]>('/api/models');
      setModels(data);
    } catch (err) {
      console.error('Failed to fetch models:', err);
    }
  };

  useEffect(() => {
    fetchRuns();
    fetchModels();
    
    // Poll for updates
    const interval = setInterval(() => {
      fetchRuns();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container" style={{ padding: '2rem' }}>
      {/* Header */}
      <header style={{ marginBottom: '2rem' }}>
        <div className="flex items-center justify-between">
          <div>
            <h1 style={{ 
              background: 'var(--color-accent-gradient)', 
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              marginBottom: '0.5rem'
            }}>
              Polymarket ML Trader
            </h1>
            <p style={{ color: 'var(--color-text-muted)' }}>
              Machine learning trading system for prediction markets
            </p>
          </div>
          <div className="flex gap-sm">
            <span className="badge badge-success">● Connected</span>
          </div>
        </div>
      </header>

      {/* Stats Overview */}
      <div className="grid grid-cols-4 mb-lg">
        <StatsCard 
          label="Active Runs" 
          value={runs.filter(r => r.status === 'running').length.toString()}
        />
        <StatsCard 
          label="Total Models" 
          value={models.length.toString()}
        />
        <StatsCard 
          label="Best Sharpe" 
          value="1.45"
          positive={true}
        />
        <StatsCard 
          label="Win Rate" 
          value="67%"
          change="5%"
          positive={true}
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-sm mb-lg">
        {(['training', 'simulation', 'models'] as const).map((tab) => (
          <button
            key={tab}
            className={`btn ${activeTab === tab ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Main Content */}
      {activeTab === 'training' && (
        <div className="grid grid-cols-2" style={{ gap: '1.5rem' }}>
          <div className="flex flex-col gap-lg">
            <TrainingPanel onTrainingStart={fetchRuns} />
            <TrainingRunsList runs={runs} onRefresh={fetchRuns} />
          </div>
          <div className="flex flex-col gap-lg">
            <LearningCurveChart data={learningData} />
            <ModelsList models={models} />
          </div>
        </div>
      )}

      {activeTab === 'simulation' && (
        <div>
          <SimulationResultCard 
            result={{
              market_id: 'btc-100k',
              total_pnl: 1234.56,
              total_return: 0.1234,
              sharpe_ratio: 1.45,
              max_drawdown: 0.082,
              win_rate: 0.67,
              total_trades: 156,
            }}
          />
        </div>
      )}

      {activeTab === 'models' && (
        <div className="grid grid-cols-2" style={{ gap: '1.5rem' }}>
          <ModelsList models={models} />
        </div>
      )}
    </div>
  );
}

export default App;
