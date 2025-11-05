import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch
print(torch.cuda.is_available())
from dataloader.data_loader import *
from policy.policy import *
# from trainer.trainer import *
from stable_baselines3 import PPO
from trainer.irl_trainer import *
from torch_geometric.loader import DataLoader

PATH_DATA = f'./dataset/'

def train_predict(args, predict_dt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    train_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                        train_start_date=args.train_start_date, train_end_date=args.train_end_date,
                                        mode="train")
    val_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                      val_start_date=args.val_start_date, val_end_date=args.val_end_date,
                                      mode="val")
    test_dataset = AllGraphDataSampler(base_dir=data_dir, date=True,
                                       test_start_date=args.test_start_date, test_end_date=args.test_end_date,
                                       mode="test")
    train_loader_all = DataLoader(train_dataset, batch_size=len(train_dataset), pin_memory=True, collate_fn=lambda x: x,
                                  drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True)
    print(len(train_loader), len(val_loader), len(test_loader))

    # create or load model
    env_init = create_env_init(args, dataset=train_dataset)
    if args.policy == 'MLP':
        if getattr(args, 'resume_model_path', None) and os.path.exists(args.resume_model_path):
            print(f"Loading PPO model from {args.resume_model_path}")
            model = PPO.load(args.resume_model_path, env=env_init, device=args.device)
        else:
            model = PPO(policy='MlpPolicy',
                        env=env_init,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device=args.device)
    elif args.policy == 'HGAT':
        policy_kwargs = dict(
            last_layer_dim_pi=args.num_stocks,  # Should equal num_stocks for proper initialization
            last_layer_dim_vf=args.num_stocks,
            n_head=8,
            hidden_dim=128,
            no_ind=(not args.ind_yn),
            no_neg=(not args.neg_yn),
        )
        if getattr(args, 'resume_model_path', None) and os.path.exists(args.resume_model_path):
            print(f"Loading PPO model from {args.resume_model_path}")
            model = PPO.load(args.resume_model_path, env=env_init, device=args.device)
        else:
            model = PPO(policy=HGATActorCriticPolicy,
                        env=env_init,
                        policy_kwargs=policy_kwargs,
                        **PPO_PARAMS,
                        seed=args.seed,
                        device=args.device)
    train_model_and_predict(model, args, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transaction ..")
    parser.add_argument("-device", "-d", default="cuda:0", help="gpu")
    parser.add_argument("-model_name", "-nm", default="SmartFolio", help="模型名称")
    parser.add_argument("-market", "-mkt", default="hs300", help="股票市场")
    parser.add_argument("-horizon", "-hrz", default="1", help="预测距离")
    parser.add_argument("-relation_type", "-rt", default="hy", help="股票关系类型")
    parser.add_argument("-ind_yn", "-ind", default="y", help="是否加入行业关系图")
    parser.add_argument("-pos_yn", "-pos", default="y", help="是否加入动量关系图")
    parser.add_argument("-neg_yn", "-neg", default="y", help="是否加入反转关系图")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="是否加入多奖励学习")
    parser.add_argument("-policy", "-p", default="MLP", help="策略网络")
    # continual learning / resume
    parser.add_argument("--resume_model_path", default=None, help="Path to previously saved PPO model to resume from")
    parser.add_argument("--reward_net_path", default=None, help="Path to saved IRL reward network state_dict to resume from")
    parser.add_argument("--fine_tune_steps", type=int, default=5000, help="Timesteps for monthly fine-tuning when resuming")
    parser.add_argument("--save_dir", default="./checkpoints", help="Directory to save trained models")
    # Training hyperparameters
    parser.add_argument("--irl_epochs", type=int, default=50, help="Number of IRL training epochs")
    parser.add_argument("--rl_timesteps", type=int, default=10000, help="Number of RL timesteps for training")
    parser.add_argument("--ga_generations", type=int, default=30, help="Number of GA generations for expert generation")
    # Expert generation strategy
    parser.add_argument("--expert_type", type=str, default="ga", 
                        choices=["ga", "ensemble", "heuristic"],
                        help="Expert generation strategy: ga=Genetic Algorithm, ensemble=Hybrid Ensemble (MV+RP+MinVar+MaxSharpe), heuristic=Original")
    # Risk-adaptive reward parameters
    parser.add_argument("--risk_score", type=float, default=0.5, help="User risk score: 0=conservative, 1=aggressive")
    parser.add_argument("--dd_base_weight", type=float, default=1.0, help="Base weight for drawdown penalty")
    parser.add_argument("--dd_risk_factor", type=float, default=1.0, help="Risk factor k in β_dd(ρ) = β_base*(1+k*(1-ρ))")
    args = parser.parse_args()

    # debug 用参数设置
    args.model_name = 'SmartFolio'
    args.market = 'hs300'
    args.relation_type = 'hy'
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.train_start_date = '2020-01-02'
    args.train_end_date = '2023-01-01'
    args.val_start_date = '2023-01-02'
    args.val_end_date = '2023-12-29'
    args.test_start_date = '2024-01-02'
    args.test_end_date = '2024-12-30'
    args.batch_size = 64
    args.max_epochs = 10
    args.seed = 123
    args.input_dim = 6
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True
    args.use_ga_expert = True  # Use GA for expert generation (set False for original heuristic)
    # Training hyperparameters (can be overridden via command line)
    args.irl_epochs = getattr(args, 'irl_epochs', 50)
    args.rl_timesteps = getattr(args, 'rl_timesteps', 10000)
    args.ga_generations = getattr(args, 'ga_generations', 30)
    # Risk-adaptive reward parameters
    args.risk_score = getattr(args, 'risk_score', 0.5)
    args.dd_base_weight = getattr(args, 'dd_base_weight', 1.0)
    args.dd_risk_factor = getattr(args, 'dd_risk_factor', 1.0)
    # ensure save dir
    os.makedirs(args.save_dir, exist_ok=True)

    if args.market == 'hs300':
        args.num_stocks = 102
    elif args.market == 'zz500':
        args.num_stocks = 80
    elif args.market == 'nd100':
        args.num_stocks = 84
    elif args.market == 'sp500':
        args.num_stocks = 472
    elif args.market == 'custom':
        # Auto-detect num_stocks from a sample pickle file
        data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        if sample_files:
            import pickle
            sample_path = os.path.join(data_dir, sample_files[0])
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            args.num_stocks = sample_data['features'].shape[1]
            print(f"Auto-detected num_stocks for custom market: {args.num_stocks}")
        else:
            raise ValueError(f"No pickle files found in {data_dir} to determine num_stocks")
    else:
        # Generic fallback for unknown markets
        data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        if os.path.exists(data_dir):
            import pickle
            sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
            if sample_files:
                sample_path = os.path.join(data_dir, sample_files[0])
                with open(sample_path, 'rb') as f:
                    sample_data = pickle.load(f)
                args.num_stocks = sample_data['features'].shape[1]
                print(f"Auto-detected num_stocks for {args.market} market: {args.num_stocks}")
            else:
                raise ValueError(f"No pickle files found in {data_dir} to determine num_stocks")
        else:
            raise ValueError(f"Unknown market {args.market} and data directory {data_dir} does not exist")

    trained_model = train_predict(args, predict_dt='2025-02-05')
    # save PPO model checkpoint
    try:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(args.save_dir, f"ppo_{args.policy.lower()}_{args.market}_{ts}")
        # train_predict currently returns None; saving env-attached model is handled inside trainer
        # If we had a handle, we could save here. Keep path ready for future.
        print(f"Training run complete. To save PPO model, call model.save('{out_path}') where model is your PPO instance.")
    except Exception as e:
        print(f"Skip saving PPO model here: {e}")

    print(1)




