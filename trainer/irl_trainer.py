import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch_geometric.data import DataLoader

from env.portfolio_env import *


class RewardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        state = state.squeeze()
        action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=1)
        return self.fc(x)


# 最大熵IRL训练器
class MaxEntIRL:
    def __init__(self, reward_net, expert_data, lr=1e-3):
        self.reward_net = reward_net
        self.expert_data = expert_data
        self.optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr)

    def train(self, agent_env, model, num_epochs=50, batch_size=32, device='cuda:0'):
        for epoch in range(num_epochs):
            # 生成代理轨迹
            agent_trajectories = self._sample_trajectories(agent_env, model, batch_size=batch_size, device=device)

            # 计算专家和代理的奖励差异
            expert_rewards = self._calculate_rewards(self.expert_data, device)
            agent_rewards = self._calculate_rewards(agent_trajectories, device)

            # 最大熵IRL损失
            loss = -(expert_rewards.mean() - torch.logsumexp(agent_rewards, dim=0))

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"Train IRL Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    def _sample_trajectories(self, env, model, batch_size, device):
        trajectories = []
        obs = env.reset()
        # obs is now 1D flattened: [num_envs, obs_len]
        # For DummyVecEnv, obs shape is [1, obs_len]
        
        # Determine num_stocks from action space
        num_stocks = env.action_space.nvec[0]  # First dimension of MultiDiscrete action space
        
        for _ in range(batch_size):
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = env.step(action)

            # 转换为 Multi-Hot 编码
            action_multi_hot = np.zeros(num_stocks)
            for i in range(action.shape[1]):
                action_multi_hot[action[:, i]] = 1

            trajectories.append((obs.copy(), action_multi_hot))
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
            reward = self.reward_net(state_tensor, action_tensor)
            rewards.append(reward)
        return torch.cat(rewards)


class MultiRewardNetwork(nn.Module):
    def __init__(self, input_dim, num_stocks, hidden_dim=64,
                 ind_yn=False, pos_yn=False, neg_yn=False):
        super().__init__()
        self.num_stocks = num_stocks
        self.input_dim = input_dim
        
        # Calculate dimensions for flattened format
        # Observation: [ind_matrix, pos_matrix, neg_matrix, features]
        # Each matrix is num_stocks x num_stocks, features is num_stocks x input_dim
        self.feature_dims = {
            'ind': num_stocks * num_stocks if ind_yn else 0,
            'pos': num_stocks * num_stocks if pos_yn else 0,
            'neg': num_stocks * num_stocks if neg_yn else 0,
            'base': num_stocks * input_dim  # Flattened features
        }

        # 动态构建编码器
        self.encoders = nn.ModuleDict()
        for feat, dim in self.feature_dims.items():
            if dim > 0:
                # For flattened input, we process the entire flattened vector + action
                self.encoders[feat] = nn.Sequential(
                    nn.Linear(dim + num_stocks, hidden_dim),  # +num_stocks for action (multi-hot)
                    nn.ReLU()
                )

        # 奖励权重参数
        active_feats = [k for k, v in self.feature_dims.items() if v > 0]
        self.num_rewards = len(active_feats)
        self.weights = nn.Parameter(torch.ones(self.num_rewards))

    def forward(self, state, action):
        # state is flattened: [batch, obs_len] where obs_len = 3*num_stocks^2 + num_stocks*input_dim
        # action is multi-hot: [batch, num_stocks]
        
        # Handle single sample case
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # 分割特征
        ptr = 0
        features = {}
        for feat, dim in self.feature_dims.items():
            if dim > 0:
                features[feat] = state[..., ptr:ptr + dim]  # [B, dim]
                ptr += dim

        # 特征-动作融合
        rewards = []
        for i, (feat, data) in enumerate(features.items()):
            # data: [B, dim], action: [B, num_stocks]
            fused = torch.cat([data, action], dim=-1)  # [B, dim + num_stocks]
            encoded = self.encoders[feat](fused)  # [B, hidden_dim]
            rewards.append(encoded.mean(dim=-1, keepdim=True))  # [B, 1]

        # 加权奖励
        weighted = sum(w * r for w, r in zip(F.softmax(self.weights, dim=0), rewards))
        return weighted


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


# 用于创建占位环境，后续使用model.set_env()进行更新
def create_env_init(args, dataset=None, data_loader=None):
    if data_loader is None:
        data_loader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True, collate_fn=lambda x: x,
                                 drop_last=True)
    for batch_idx, data in enumerate(data_loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
        env = StockPortfolioEnv(args=args, corr=corr, ts_features=ts_features, features=features,
                                ind=ind, pos=pos, neg=neg,
                                returns=labels, pyg_data=pyg_data, device=args.device,
                                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn)
        env.seed(seed=args.seed)
        env, _ = env.get_sb_env()
        print("占位环境创建完成")
        return env


PPO_PARAMS = {
        "n_steps": 1024,
        "ent_coef": 0.005,
        "learning_rate": 1e-4,
        "batch_size": 128,
        "gamma": 0.5,
        "tensorboard_log": "./logs",
    }


def model_predict(args, model, test_loader):
    # 读取指数 benchmark 数据，用于计算信息系数 IR
    df_benchmark = pd.read_csv(f"../dataset/index_data/{args.market}_index_2024.csv")
    df_benchmark = df_benchmark[(df_benchmark['datetime'] >= args.test_start_date) &
                                (df_benchmark['datetime'] <= args.test_end_date)]
    benchmark_return = df_benchmark['daily_return']
    for batch_idx, data in enumerate(test_loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
        env_test = StockPortfolioEnv(args=args, corr=corr, ts_features=ts_features, features=features,
                                     ind=ind, pos=pos, neg=neg,
                                     returns=labels, pyg_data=pyg_data, benchmark_return=benchmark_return,
                                     mode="test", ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn)
        env_test, obs_test = env_test.get_sb_env()
        env_test.reset()
        max_step = len(labels)
        for i in range(max_step):
            action, _states = model.predict(obs_test)
            obs_test, rewards, dones, info = env_test.step(action)
            if dones[0]:
                break


def train_model_and_predict(model, args, train_loader, val_loader, test_loader):
    # --- 生成专家轨迹 ---
    from gen_data.generate_expert import generate_expert_trajectories
    expert_trajectories = generate_expert_trajectories(
        args, train_loader.dataset, num_trajectories=10000
    )

    # --- 初始化IRL奖励网络 ---
    # With flattened observation format:
    # obs_len = 3 * num_stocks^2 + num_stocks * input_dim
    obs_len = 3 * args.num_stocks * args.num_stocks + args.num_stocks * args.input_dim
    
    if not args.multi_reward:
        reward_net = RewardNetwork(input_dim=obs_len+1).to(args.device)
        irl_trainer = MaxEntIRL(reward_net, expert_trajectories, lr=1e-4)
    else:
        reward_net = MultiRewardNetwork(input_dim=args.input_dim,
                                        num_stocks=args.num_stocks,
                                        ind_yn=args.ind_yn,
                                        pos_yn=args.pos_yn,
                                        neg_yn=args.neg_yn).to(args.device)
        irl_trainer = MaxEntIRL(reward_net, expert_trajectories, lr=1e-4)

    # --- train ---
    env_train = create_env_init(args, data_loader=train_loader)
    for i in range(args.max_epochs):
        # 1. 训练IRL奖励函数
        irl_trainer.train(env_train, model, num_epochs=5,
                          batch_size=args.batch_size, device=args.device)  # 假设env_train是当前RL环境
        print("reward net train over.")

        # 2. 更新RL环境使用新奖励函数
        for batch_idx, data in enumerate(train_loader):
            corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
            env_train = StockPortfolioEnv(
                args=args, corr=corr, ts_features=ts_features, features=features,
                ind=ind, pos=pos, neg=neg,
                returns=labels, pyg_data=pyg_data, reward_net=reward_net, device=args.device,
                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn
            )
            env_train.seed(seed=args.seed)
            env_train, _ = env_train.get_sb_env()
            model.set_env(env_train)

            # 3. 训练RL代理
            trained_model = model.learn(total_timesteps=10000)
            # 评估训练后的模型
            mean_reward, std_reward = evaluate_policy(model, env_train, n_eval_episodes=1)
            print(f"平均奖励: {mean_reward}")
            model_predict(args, trained_model, test_loader)
        return trained_model