import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch_geometric.data import DataLoader

from env.portfolio_env import *
from trainer.evaluation_utils import (
    aggregate_metric_records,
    apply_promotion_gate,
    create_metric_record,
    persist_metrics,
)
from utils.ticker_mapping import (
    get_ticker_mapping_for_period,
    load_ticker_mapping,
)
from gen_data.gen_expert_ensemble import (
    generate_expert_trajectories,
    load_expert_trajectories,
    save_expert_trajectories,
)


class RewardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action, wealth_info=None):
        # wealth_info is ignored in this simple network (for compatibility)
        state = state.squeeze()
        action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=1)
        return self.fc(x)


# Maximum entropy IRL trainer
class MaxEntIRL:
    def __init__(self, reward_net, expert_data, lr=1e-3, risk_score: float = 0.5):
        self.reward_net = reward_net
        self.expert_data = expert_data
        self.optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr)
        self.risk_score = float(risk_score)

    def train(self, agent_env, model, num_epochs=50, batch_size=32, device='cuda:0'):
        for epoch in range(num_epochs):
            # Generate agent trajectories
            agent_trajectories = self._sample_trajectories(agent_env, model, batch_size=batch_size, device=device)

            # Compute the reward gap between expert and agent rollouts
            expert_rewards = self._calculate_rewards(self.expert_data, device)
            agent_rewards = self._calculate_rewards(agent_trajectories, device)

            # Maximum entropy IRL loss
            # Original: loss = -(expert_rewards.mean() - torch.logsumexp(agent_rewards, dim=0))
            # Reformulated to always show meaningful values:
            expert_mean = expert_rewards.mean()
            agent_logsumexp = torch.logsumexp(agent_rewards, dim=0)
            
            # The actual loss (can be negative, which is OK)
            loss = -(expert_mean - agent_logsumexp)
            
            # For logging: show the gap (always interpretable)
            reward_gap = expert_mean - agent_logsumexp

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Better logging: show both loss and interpretable metrics
            print(f"Train IRL Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, "
                  f"Expert: {expert_mean.item():.4f}, Agent: {agent_logsumexp.item():.4f}, "
                  f"Gap: {reward_gap.item():.4f}")

    def _sample_trajectories(self, env, model, batch_size, device):
        trajectories = []
        obs = env.reset()
        # obs is now 1D flattened: [num_envs, obs_len]
        # For DummyVecEnv, obs shape is [1, obs_len]
        
        # Determine num_stocks from action space (now Box, not MultiDiscrete)
        num_stocks = env.action_space.shape[0]  # Box space has shape attribute
        
        for _ in range(batch_size):
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = env.step(action)

            # action is now continuous weights [num_envs, num_stocks]
            # Normalize to ensure sum to 1 (apply softmax)
            action_scores = action[0]  # Get first env's action
            exp_actions = np.exp(action_scores - np.max(action_scores))
            action_weights = exp_actions / exp_actions.sum()

            trajectories.append((obs.copy(), action_weights))
            obs = next_obs
            if done:
                obs = env.reset()
        return trajectories

    def _calculate_rewards(self, trajectories, device):
        rewards = []
        for state, action in trajectories:
            # Handle different state shapes
            if isinstance(state, np.ndarray):
                # If state has shape [1, obs_len], squeeze to [obs_len]
                if state.ndim > 1 and state.shape[0] == 1:
                    state = state.squeeze(0)
                # If state already has correct shape, keep as is
            state_tensor = torch.FloatTensor(state).to(device)
            action_tensor = torch.FloatTensor(action).to(device)
            
            # For expert trajectories, we don't have wealth info, so pass None
            # The reward network will handle this appropriately
            reward = self.reward_net(
                state_tensor,
                action_tensor,
                wealth_info=None,
                risk_score=self.risk_score,
            )
            rewards.append(reward)
        return torch.cat(rewards)


class MultiRewardNetwork(nn.Module):
    def __init__(self, input_dim, num_stocks, hidden_dim=64,
                 ind_yn=False, pos_yn=False, neg_yn=False,
                 use_drawdown=True, dd_kappa=0.05, dd_tau=0.01,
                 dd_base_weight=1.0, dd_risk_factor=1.0):
        super().__init__()
        self.num_stocks = num_stocks
        self.input_dim = input_dim
        self.use_drawdown = use_drawdown
        self.dd_kappa = dd_kappa  # Ignore drawdowns below this threshold
        self.dd_tau = dd_tau  # Smoothness parameter for SoftPlus
        self.dd_base_weight = dd_base_weight  # β_base for drawdown
        self.dd_risk_factor = dd_risk_factor  # k in β_dd(ρ) = β_base * (1 + k*(1-ρ))
        
        # Risk-adaptive kappa (optional, can be enabled later)
        self.dd_kappa_max = 0.10  # Conservative users tolerate less drawdown
        self.dd_kappa_min = 0.02  # Aggressive users tolerate more drawdown
        
        # Calculate dimensions for flattened format
        # Observation: [ind_matrix, pos_matrix, neg_matrix, features]
        # Each matrix is num_stocks x num_stocks, features is num_stocks x input_dim
        self.feature_dims = {
            'ind': num_stocks * num_stocks if ind_yn else 0,
            'pos': num_stocks * num_stocks if pos_yn else 0,
            'neg': num_stocks * num_stocks if neg_yn else 0,
            'base': num_stocks * input_dim  # Flattened features
        }

        # Build encoders dynamically based on enabled relations
        self.encoders = nn.ModuleDict()
        for feat, dim in self.feature_dims.items():
            if dim > 0:
                # For flattened input, we process the entire flattened vector + action
                self.encoders[feat] = nn.Sequential(
                    nn.Linear(dim + num_stocks, hidden_dim),  # +num_stocks for action (multi-hot)
                    nn.ReLU()
                )

        # Drawdown penalty encoder
        if self.use_drawdown:
            self.dd_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim),  # Input: [W_current, W_peak]
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        # Reward weight parameters (one per enabled feature block, plus drawdown)
        active_feats = [k for k, v in self.feature_dims.items() if v > 0]
        self.num_rewards = len(active_feats)
        if self.use_drawdown:
            self.num_rewards += 1  # Add drawdown component
        self.weights = nn.Parameter(torch.ones(self.num_rewards))

    def forward(self, state, action, wealth_info=None, risk_score=None):
        # state is flattened: [batch, obs_len] where obs_len = 3*num_stocks^2 + num_stocks*input_dim
        # action is multi-hot: [batch, num_stocks]
        # wealth_info: [batch, 2] containing [W_current, W_peak]
        # risk_score: [batch, 1] or scalar, ρ ∈ [0, 1] where 0=conservative, 1=aggressive
        
        # Handle single sample case
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if wealth_info is not None and wealth_info.dim() == 1:
            wealth_info = wealth_info.unsqueeze(0)
        
        batch_size = state.shape[0]
        
        # Default risk score if not provided (moderate = 0.5)
        if risk_score is None:
            risk_score = torch.ones(batch_size, 1, device=state.device) * 0.5
        elif isinstance(risk_score, (int, float)):
            risk_score = torch.ones(batch_size, 1, device=state.device) * risk_score
        elif risk_score.dim() == 0:
            risk_score = risk_score.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif risk_score.dim() == 1:
            risk_score = risk_score.unsqueeze(1)
        
        # Split flattened features back into logical blocks
        ptr = 0
        features = {}
        for feat, dim in self.feature_dims.items():
            if dim > 0:
                features[feat] = state[..., ptr:ptr + dim]  # [B, dim]
                ptr += dim

        # Feature-action fusion (IRL component)
        irl_rewards = []
        for i, (feat, data) in enumerate(features.items()):
            # data: [B, dim], action: [B, num_stocks]
            fused = torch.cat([data, action], dim=-1)  # [B, dim + num_stocks]
            encoded = self.encoders[feat](fused)  # [B, hidden_dim]
            irl_rewards.append(encoded.mean(dim=-1, keepdim=True))  # [B, 1]

        # Combine IRL rewards
        R_irl = sum(w * r for w, r in zip(F.softmax(self.weights[:-1] if self.use_drawdown else self.weights, dim=0), irl_rewards))

        # Drawdown penalty component (risk-adaptive)
        if self.use_drawdown and wealth_info is not None:
            # wealth_info: [B, 2] = [W_current, W_peak]
            W_current = wealth_info[:, 0:1]  # [B, 1]
            W_peak = wealth_info[:, 1:2]  # [B, 1]
            
            # Calculate drawdown: dd = (W_peak - W_current) / max(W_peak, epsilon)
            epsilon = 1e-8
            dd = (W_peak - W_current) / torch.clamp(W_peak, min=epsilon)  # [B, 1]
            
            # Risk-adaptive kappa (optional): κ(ρ) = κ_max - (κ_max - κ_min) * ρ
            # Conservative (ρ=0) → high κ (ignore small drawdowns)
            # Aggressive (ρ=1) → low κ (penalize even small drawdowns)
            kappa_adaptive = self.dd_kappa_max - (self.dd_kappa_max - self.dd_kappa_min) * risk_score
            
            # Smooth penalty: -SoftPlus((dd - kappa) / tau)
            dd_scaled = (dd - kappa_adaptive) / self.dd_tau
            R_dd = -F.softplus(dd_scaled)  # [B, 1]
            
            # Risk-adaptive weight: β_dd(ρ) = β_base * (1 + k * (1 - ρ))
            # Conservative (ρ=0) → β_dd = β_base * (1 + k) = 2x penalty
            # Aggressive (ρ=1) → β_dd = β_base * 1 = baseline penalty
            beta_dd = self.dd_base_weight * (1 + self.dd_risk_factor * (1 - risk_score))
            
            # Total reward: R_total = R_irl + β_dd(ρ) * R_dd
            return R_irl + beta_dd * R_dd
        
        return R_irl


def process_data(data_dict, device="cuda:0"):
    corr = data_dict['corr'].to(device).squeeze()
    ts_features = data_dict['ts_features'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    industry_matrix = data_dict['industry_matrix'].to(device).squeeze()
    pos_matrix = data_dict['pos_matrix'].to(device).squeeze()
    neg_matrix = data_dict['neg_matrix'].to(device).squeeze()
    pyg_data = data_dict['pyg_data'].to(device)
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    return corr, ts_features, features,\
           industry_matrix, pos_matrix, neg_matrix,\
           labels, pyg_data, mask


def resolve_expert_cache_path(args, risk_score):
    cache_setting = getattr(args, "expert_cache_path", None)
    if not cache_setting:
        return None
    cache_setting = os.path.expanduser(cache_setting)

    def _default_filename():
        return (
            f"experts_{args.market}_{args.train_start_date}_{args.train_end_date}_"
            f"{args.horizon}_{args.relation_type}_risk{risk_score:.2f}.pkl"
        )

    is_dir_hint = cache_setting.endswith(os.sep) or os.path.splitext(cache_setting)[1] == ""
    if os.path.isdir(cache_setting) or is_dir_hint:
        base_dir = cache_setting.rstrip(os.sep)
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, _default_filename())

    base_dir = os.path.dirname(cache_setting)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    return cache_setting


# Create a placeholder environment that can later be swapped via model.set_env()
def create_env_init(args, dataset=None, data_loader=None):
    if data_loader is None:
        data_loader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True, collate_fn=lambda x: x,
                                 drop_last=True)
    for batch_idx, data in enumerate(data_loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
        env = StockPortfolioEnv(args=args, corr=corr, ts_features=ts_features, features=features,
                                ind=ind, pos=pos, neg=neg,
                                returns=labels, pyg_data=pyg_data, device=args.device,
                                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn,
                                risk_profile=getattr(args, 'risk_profile', None))
        env.seed(seed=args.seed)
        env, _ = env.get_sb_env()
        print("Environment created")
        return env


PPO_PARAMS = {
        "n_steps": 256,  # Reduced from 1024 to prevent OOM with large obs space
        "ent_coef": 0.005,
        "learning_rate": 1e-4,
        "batch_size": 64,  # Reduced from 128 to match n_steps/2
        "gamma": 0.99,
        "tensorboard_log": "./logs",
    }


def model_predict(args, model, test_loader, split: str = "test"):
    """Evaluate a model on the provided loader and persist metrics."""

    df_benchmark = pd.read_csv(f"./dataset/index_data/{args.market}_index.csv")
    df_benchmark = df_benchmark[(df_benchmark['datetime'] >= args.test_start_date) &
                                (df_benchmark['datetime'] <= args.test_end_date)]
    benchmark_return = df_benchmark['daily_return']
    
    # Load ticker mappings for the test period
    ticker_map = get_ticker_mapping_for_period(
        args.market,
        args.test_start_date,
        args.test_end_date,
        base_dir="dataset_default"
    )

    records = []
    env_snapshots = []
    all_weights_data = []  # Collect all weights for CSV export

    for batch_idx, data in enumerate(test_loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
        env_test = StockPortfolioEnv(
            args=args,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=labels,
            pyg_data=pyg_data,
            benchmark_return=benchmark_return,
            mode="test",
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            risk_profile=getattr(args, 'risk_profile', None),
        )

        obs_test = env_test.reset()
        max_step = len(labels)

        for _ in range(max_step):
            action, _states = model.predict(obs_test, deterministic=False)
            obs_test, reward, done, info = env_test.step(action)
            if done:
                break

        arr, avol, sharpe, mdd, cr, ir = env_test.evaluate()
        metrics = {
            "arr": arr,
            "avol": avol,
            "sharpe": sharpe,
            "mdd": mdd,
            "cr": cr,
            "ir": ir,
        }
        record = create_metric_record(args, split, metrics, batch_idx)
        records.append(record)
        env_snapshots.append((env_test, record["run_id"]))
        
        # Collect weights data from the environment
        weights_array = env_test.get_weights_history()
        print("weights array", weights_array)
        if weights_array.size > 0:
            num_steps, num_stocks = weights_array.shape
            date_keys = list(ticker_map.keys()) if ticker_map else []
            step_labels = getattr(env_test, "dates", list(range(num_steps)))

            for step_idx in range(num_steps):
                weights = weights_array[step_idx]
                date_label = date_keys[step_idx] if step_idx < len(date_keys) else None

                tickers = None
                if date_label:
                    candidate = ticker_map.get(date_label)
                    if candidate and len(candidate) == num_stocks:
                        tickers = candidate

                if tickers is None:
                    tickers = [f"stock_{i}" for i in range(num_stocks)]
                    if date_label is None:
                        date_val = step_labels[step_idx] if step_idx < len(step_labels) else step_idx
                        date_label = f"step_{date_val}"

                step_value = step_labels[step_idx] if step_idx < len(step_labels) else step_idx

                for ticker, weight in zip(tickers, weights):
                    if weight > 0.0001:  # Only save non-negligible allocations
                        all_weights_data.append({
                            'run_id': record["run_id"],
                            'batch': batch_idx,
                            'date': date_label,
                            'step': step_value,
                            'ticker': ticker,
                            'weight': weight,
                            'weight_pct': weight * 100
                        })
        
    log_info = persist_metrics(records, env_snapshots, args, split)
    summary = aggregate_metric_records(records)
    
    # Save portfolio weights to CSV
    if all_weights_data and log_info and 'log_dir' in log_info:
        log_dir = log_info['log_dir']
        timestamp_label = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        weights_csv_path = os.path.join(log_dir, f"{split}_weights_{timestamp_label}.csv")
        
        df_weights = pd.DataFrame(all_weights_data)
        df_weights.to_csv(weights_csv_path, index=False)
        print(f"Saved portfolio weights to: {weights_csv_path}")
        log_info['weights_csv'] = weights_csv_path
        
        # Also save a summary showing average weights per ticker
        summary_weights = df_weights.groupby('ticker').agg({
            'weight': ['mean', 'std', 'min', 'max', 'count']
        }).round(6)
        summary_weights.columns = ['avg_weight', 'std_weight', 'min_weight', 'max_weight', 'num_days']
        summary_weights = summary_weights.sort_values('avg_weight', ascending=False)
        
        summary_path = os.path.join(log_dir, f"{split}_weights_summary_{timestamp_label}.csv")
        summary_weights.to_csv(summary_path)
        print(f"Saved weights summary to: {summary_path}")
        log_info['weights_summary'] = summary_path

    if summary:
        print(f"Evaluation summary for split='{split}': {summary}")

    return {
        "records": records,
        "summary": summary,
        "log": log_info,
    }


def train_model_and_predict(model, args, train_loader, val_loader, test_loader):
    print("\n" + "=" * 70)
    print("Using Hybrid Ensemble Expert Generation")
    print("=" * 70)
    profile = getattr(args, "risk_profile", None)
    risk_score_value = profile.get("risk_score", getattr(args, "risk_score", 0.5)) if profile else getattr(args, "risk_score", 0.5)
    cache_file = resolve_expert_cache_path(args, risk_score_value)
    expert_trajectories = None
    if cache_file and os.path.exists(cache_file):
        try:
            print(f"Loading cached expert trajectories from {cache_file}")
            expert_trajectories = load_expert_trajectories(cache_file)
        except Exception as exc:
            print(f"Warning: failed to load cached expert trajectories ({exc}); regenerating.")

    if expert_trajectories is None:
        expert_trajectories = generate_expert_trajectories(
            args,
            train_loader.dataset,
            num_trajectories=100,
            risk_profile=profile,
        )
        if cache_file:
            try:
                save_expert_trajectories(expert_trajectories, cache_file)
            except Exception as exc:
                print(f"Warning: could not save expert trajectories to {cache_file}: {exc}")

    # --- Initialize the IRL reward network ---
    # With flattened observation format:
    # obs_len = 3 * num_stocks^2 + num_stocks * input_dim
    obs_len = 3 * args.num_stocks * args.num_stocks + args.num_stocks * args.input_dim
    
    if not args.multi_reward:
        reward_net = RewardNetwork(input_dim=obs_len+1).to(args.device)
    else:
        reward_net = MultiRewardNetwork(
            input_dim=args.input_dim,
            num_stocks=args.num_stocks,
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            dd_base_weight=getattr(args, 'dd_base_weight', 1.0),
            dd_risk_factor=getattr(args, 'dd_risk_factor', 1.0)
        ).to(args.device)
    # Optional: resume reward network
    if getattr(args, 'reward_net_path', None) and os.path.exists(args.reward_net_path):
        try:
            print(f"Loading reward network from {args.reward_net_path}")
            state = torch.load(args.reward_net_path, map_location=args.device)
            reward_net.load_state_dict(state)
        except Exception as e:
            print(f"Warning: failed to load reward net: {e}")
    irl_trainer = MaxEntIRL(reward_net, expert_trajectories, lr=1e-4, risk_score=risk_score_value)

    # --- train ---
    env_train = create_env_init(args, data_loader=train_loader)
    for i in range(args.max_epochs):
        print(f"\n=== Epoch {i+1}/{args.max_epochs} ===")
        # 1. Train the IRL reward function
        irl_epochs = getattr(args, 'irl_epochs', 50)
        irl_trainer.train(env_train, model, num_epochs=irl_epochs,
                          batch_size=args.batch_size, device=args.device)
        print("reward net train over.")

        # 2. Update the RL environment with the new reward function (only the first batch)
        for batch_idx, data in enumerate(train_loader):
            if batch_idx > 0:  # Only process first batch
                break
            corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
            env_train = StockPortfolioEnv(
                args=args, corr=corr, ts_features=ts_features, features=features,
                ind=ind, pos=pos, neg=neg,
                returns=labels, pyg_data=pyg_data, reward_net=reward_net, device=args.device,
                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn,
                risk_profile=getattr(args, 'risk_profile', None)
            )
            env_train.seed(seed=args.seed)
            env_train, _ = env_train.get_sb_env()
            model.set_env(env_train)
            rl_timesteps = getattr(args, 'rl_timesteps', 10000)
            # 3. Train the RL agent
            print(f"Training RL agent for {rl_timesteps} timesteps...")
            model = model.learn(total_timesteps=rl_timesteps)
            # Optional evaluation is logged inside the environment callbacks
            mean_reward, std_reward = evaluate_policy(model, env_train, n_eval_episodes=1)
            print(f"Evaluation after RL training: Mean Reward = {mean_reward:.4f}, Std Reward = {std_reward:.4f}")
        # 4. Intermediate test evaluation to print ARR/AVOL/Sharpe/MDD/CR/IR
        print("\n=== Intermediate Test Evaluation (after RL learn) ===")
        model_predict(args, model, test_loader, split=f"epoch{i+1}_test")
    # Save reward network checkpoint
    try:
        save_dir = getattr(args, 'save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        ckpt_path = os.path.join(save_dir, f"reward_net_{args.market}_{ts}.pt")
        torch.save(reward_net.state_dict(), ckpt_path)
        print(f"Saved reward network to {ckpt_path}")
    except Exception as e:
        print(f"Warning: could not save reward net: {e}")

    # Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    final_eval = model_predict(args, model, test_loader, split="final_test")

    candidate_path = None
    try:
        save_dir = getattr(args, 'save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        policy_name = getattr(args, 'policy', 'policy').lower()
        candidate_path = os.path.join(save_dir, f"ppo_{policy_name}_{args.market}_{ts}.zip")
        model.save(candidate_path)
        print(f"Saved PPO model checkpoint to {candidate_path}")
    except Exception as e:
        print(f"Warning: could not save PPO model: {e}")

    apply_promotion_gate(args, candidate_path, final_eval.get("summary"), final_eval.get("log"))

    return model
