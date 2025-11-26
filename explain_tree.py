"""
Explain RL portfolio allocations with decision trees.

This script:
 - Loads test data using AllGraphDataSampler (matching main.py/irl_trainer.py)
 - Creates a StockPortfolioEnv for each batch
 - Loads a PPO model and collects observations and portfolio weights
 - Trains a DecisionTreeRegressor to predict weights from observations
 - Prints interpretable decision rules for selected stocks

Usage:
 python tools/explain_tree.py --model-path ./checkpoints/ppo_hgat_custom_20241230.zip --stock-index 0
"""
import os
import sys
import argparse
import numpy as np
import pickle

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch_geometric.loader import DataLoader

from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import r2_score
import joblib

# Project imports
from dataloader.data_loader import AllGraphDataSampler
from env.portfolio_env import StockPortfolioEnv
from stable_baselines3 import PPO
from utils.risk_profile import build_risk_profile


def process_data(data_dict, device="cpu"):
    """Process data dict to match irl_trainer.py format"""
    corr = data_dict['corr'].to(device).squeeze()
    ts_features = data_dict['ts_features'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    industry_matrix = data_dict['industry_matrix'].to(device).squeeze()
    pos_matrix = data_dict['pos_matrix'].to(device).squeeze()
    neg_matrix = data_dict['neg_matrix'].to(device).squeeze()
    pyg_data = data_dict['pyg_data'].to(device)
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    return corr, ts_features, features, industry_matrix, pos_matrix, neg_matrix, labels, pyg_data, mask


def map_feature_index(idx, num_stocks, input_dim):
    """Map flattened feature index to human-readable name.
    Layout: [ind_flat, pos_flat, neg_flat, features_flat]
    """
    block = num_stocks * num_stocks
    
    if idx < block:
        i = idx // num_stocks
        j = idx % num_stocks
        return f"Industry_Relation[Stock_{i}, Stock_{j}]"
    idx -= block
    
    if idx < block:
        i = idx // num_stocks
        j = idx % num_stocks
        return f"Momentum_Relation[Stock_{i}, Stock_{j}]"
    idx -= block
    
    if idx < block:
        i = idx // num_stocks
        j = idx % num_stocks
        return f"Reversal_Relation[Stock_{i}, Stock_{j}]"
    idx -= block
    
    # Features: ordered as stock0_feat0, stock0_feat1, ..., stock1_feat0, ...
    stock = idx // input_dim
    feat = idx % input_dim
    feature_names = ['Close', 'Volume', 'Return', 'Volatility', 'MA', 'Momentum']
    feat_name = feature_names[feat] if feat < len(feature_names) else f"Feature_{feat}"
    return f"Stock_{stock}_{feat_name}"


def softmax(x):
    """Softmax to convert raw action scores to portfolio weights"""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def collect_trajectories_from_loader(test_loader, model, args, device='cpu'):
    """Collect trajectories from all batches in test_loader"""
    all_obs = []
    all_weights = []
    
    for batch_idx, data in enumerate(test_loader):
        print(f"\nProcessing batch {batch_idx + 1}/{len(test_loader)}")
        
        # Process data exactly like irl_trainer.py
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=device)
        
        # Create environment
        env = StockPortfolioEnv(
            args=args,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=labels,
            pyg_data=pyg_data,
            mode="test",
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            risk_profile=getattr(args, 'risk_profile', None)
        )
        
        # Get vectorized environment
        env_vec, obs = env.get_sb_env()
        env_vec.reset()
        
        max_steps = len(labels)
        batch_obs = []
        batch_weights = []
        
        for step in range(max_steps):
            # Get action from model
            action, _ = model.predict(obs)
            
            # Convert raw action scores to portfolio weights using softmax
            action_scores = np.array(action).flatten()
            weights = softmax(action_scores)
            
            # Extract observation (handle DummyVecEnv shape [1, obs_len])
            if isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[0] == 1:
                obs_sample = obs.squeeze(0).copy()
            else:
                obs_sample = np.array(obs).flatten().copy()
            
            batch_obs.append(obs_sample)
            batch_weights.append(weights)
            
            # Step environment
            obs, rewards, dones, info = env_vec.step(action)
            
            if dones[0]:
                break
        
        env_vec.close()
        
        all_obs.extend(batch_obs)
        all_weights.extend(batch_weights)
        
        print(f"  Collected {len(batch_obs)} steps from batch {batch_idx + 1}")
    
    X = np.vstack(all_obs)
    Y = np.vstack(all_weights)
    
    return X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', default='1')
    parser.add_argument('--relation_type', default='hy')
    parser.add_argument('--model-path', required=True, help='Path to trained PPO model (.zip)')
    parser.add_argument('--test-start-date', default='2024-01-02')
    parser.add_argument('--test-end-date', default='2024-12-26')
    parser.add_argument('--max-depth', type=int, default=5, help='Max depth of decision tree')
    parser.add_argument('--stock-index', type=int, default=0, help='Which stock to explain')
    parser.add_argument('--top-k-stocks', type=int, default=5, help='Number of top stocks to explain')
    parser.add_argument('--device', default='cpu', help='Device for data loading')
    
    args = parser.parse_args()
    args.market = 'custom'
    parser.add_argument('--risk-score', type=float, default=None, help='User risk score (auto-loaded from artifacts if not provided)')
    # Auto-detect num_stocks and input_dim (matching main.py)
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    sample_files_detect = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if sample_files_detect:
        sample_path_detect = os.path.join(data_dir, sample_files_detect[0])
        with open(sample_path_detect, 'rb') as f:
            sample_data_detect = pickle.load(f)
        args.num_stocks = sample_data_detect['features'].shape[0]
        print(f"Auto-detected num_stocks for custom market: {args.num_stocks}")
    else:
        raise ValueError(f"No pickle files found in {data_dir} to determine num_stocks for custom market")
    
    # Auto-detect input_dim from sample file
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if sample_files:
        sample_path = os.path.join(data_dir, sample_files[0])
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        args.input_dim = sample_data['features'].shape[-1]
        print(f"Auto-detected input_dim: {args.input_dim}")
    else:
        args.input_dim = 6
        print("Warning: Using default input_dim=6")
    
    # Set flags for relation matrices
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.risk_score = getattr(args, 'risk_score', None)
    if args.risk_score is None:
        print(f"⚠  ERROR: risk_score not set!")
        print(f"  This should be loaded from risk_cli artifacts in main.py")
        print(f"  Proceeding with None - this may cause errors")
    
    
    print(f"\nConfiguration:")
    print(f"  Market: {args.market}")
    print(f"  Num stocks: {args.num_stocks}")
    print(f"  Input dim: {args.input_dim}")
    print(f"  Test period: {args.test_start_date} to {args.test_end_date}")
    print(f"  Model: {args.model_path}")
    
    # Load test dataset (matching main.py)
    print(f"\nLoading test data from: {data_dir}")
    test_dataset = AllGraphDataSampler(
        base_dir=data_dir,
        date=True,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        mode="test"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)} batches")
    
    # Load PPO model
    print(f"\nLoading PPO model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model = PPO.load(args.model_path, env=None, device=args.device)
    
    # Collect trajectories
    print("\nCollecting trajectories from test data...")
    X, Y = collect_trajectories_from_loader(test_loader, model, args, device=args.device)
    
    print(f"\nCollected data:")
    print(f"  Total timesteps: {X.shape[0]}")
    print(f"  Observation dim: {X.shape[1]}")
    print(f"  Number of stocks: {Y.shape[1]}")
    print(f"  Average portfolio concentration: {(Y > 0.01).sum(axis=1).mean():.1f} stocks")
    
    # Train multi-output decision tree
    print(f"\nTraining multi-output decision tree (max_depth={args.max_depth})...")
    multi_tree = DecisionTreeRegressor(max_depth=args.max_depth, random_state=42)
    multi_tree.fit(X, Y)
    Y_pred = multi_tree.predict(X)
    r2_avg = r2_score(Y, Y_pred, multioutput='uniform_average')
    print(f"Multi-output tree R² score (average): {r2_avg:.4f}")
    
    # Feature names for interpretation
    feat_names = [map_feature_index(i, args.num_stocks, args.input_dim) for i in range(X.shape[1])]
    
    # Find top-K stocks by average weight
    avg_weights = Y.mean(axis=0)
    top_k_indices = np.argsort(avg_weights)[::-1][:args.top_k_stocks]
    
    print(f"\n{'='*80}")
    print(f"Top {args.top_k_stocks} stocks by average allocation:")
    print(f"{'='*80}")
    for rank, stock_idx in enumerate(top_k_indices, 1):
        print(f"{rank}. Stock {stock_idx}: {avg_weights[stock_idx]:.4%} average weight")
    
    # Train and explain individual stock trees
    print(f"\n{'='*80}")
    print(f"Decision Rules for Top {args.top_k_stocks} Stocks")
    print(f"{'='*80}")
    
    stock_trees = {}
    for stock_idx in top_k_indices:
        print(f"\n{'='*80}")
        print(f"STOCK {stock_idx} - Average Weight: {avg_weights[stock_idx]:.4%}")
        print(f"{'='*80}")
        
        # Train tree for this stock
        tree = DecisionTreeRegressor(max_depth=args.max_depth, random_state=42)
        tree.fit(X, Y[:, stock_idx])
        r2 = r2_score(Y[:, stock_idx], tree.predict(X))
        print(f"R² score: {r2:.4f}")
        
        # Get feature importances
        importances = tree.feature_importances_
        top_features_idx = np.argsort(importances)[::-1][:10]
        
        print(f"\nTop 10 most important features:")
        for i, idx in enumerate(top_features_idx, 1):
            if importances[idx] > 0:
                print(f"  {i}. {feat_names[idx]}: {importances[idx]:.4f}")
        
        # Export decision rules
        rules = export_text(tree, feature_names=feat_names, max_depth=args.max_depth)
        print(f"\nDecision Rules:")
        print(rules)
        
        stock_trees[stock_idx] = tree
    
    # Explain specific stock if requested
    if args.stock_index not in top_k_indices:
        print(f"\n{'='*80}")
        print(f"Additional Explanation for Stock {args.stock_index}")
        print(f"{'='*80}")
        print(f"Average Weight: {avg_weights[args.stock_index]:.4%}")
        
        tree = DecisionTreeRegressor(max_depth=args.max_depth, random_state=42)
        tree.fit(X, Y[:, args.stock_index])
        r2 = r2_score(Y[:, args.stock_index], tree.predict(X))
        print(f"R² score: {r2:.4f}")
        
        rules = export_text(tree, feature_names=feat_names, max_depth=args.max_depth)
        print(f"\nDecision Rules:")
        print(rules)
        
        stock_trees[args.stock_index] = tree
    
    # Save results
    output_dir = "./explainability_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'multi_tree': multi_tree,
        'stock_trees': stock_trees,
        'feat_names': feat_names,
        'avg_weights': avg_weights,
        'top_k_indices': top_k_indices,
        'r2_multi': r2_avg,
        'X_shape': X.shape,
        'Y_shape': Y.shape
    }
    
    output_path = os.path.join(output_dir, f"explain_tree_{args.market}.joblib")
    joblib.dump(results, output_path)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
