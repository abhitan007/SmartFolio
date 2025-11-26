import os
import time
import json
import argparse
import warnings
from datetime import datetime
import calendar
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
from utils.risk_profile import build_risk_profile, get_risk_profile_description
from risk_cli import load_deployment_pipeline, run_offline_inference
PATH_DATA = f'./dataset/'


def _infer_month_dates(shard):
    """Infer month label, start, and end date strings from a manifest shard."""
    month_label = shard.get("month")
    month_start = shard.get("month_start") or shard.get("start_date") or shard.get("train_start_date")
    month_end = shard.get("month_end") or shard.get("end_date") or shard.get("train_end_date")

    # Normalise the month label
    parsed_month = None
    if month_label:
        for fmt in ("%Y-%m", "%Y-%m-%d"):
            try:
                parsed_month = datetime.strptime(month_label, fmt)
                break
            except ValueError:
                continue
    if parsed_month is None and month_start:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                parsed_month = datetime.strptime(month_start, fmt)
                month_label = parsed_month.strftime("%Y-%m")
                break
            except ValueError:
                continue

    if parsed_month and not month_start:
        month_start = parsed_month.strftime("%Y-%m-01")

    if parsed_month and not month_end and month_start:
        last_day = calendar.monthrange(parsed_month.year, parsed_month.month)[1]
        month_end = parsed_month.replace(day=last_day).strftime("%Y-%m-%d")

    if not (month_label and month_start and month_end):
        raise ValueError(f"Unable to infer complete month information from shard: {shard}")

    return month_label, month_start, month_end


def fine_tune_month(args, manifest_path="monthly_manifest.json", bookkeeping_path=None):
    """Fine-tune the PPO model on the latest unprocessed monthly shard."""
    manifest_file = manifest_path
    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f"Monthly manifest not found at {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    shards = manifest.get("monthly_shards", {})
    if not shards:
        raise ValueError("Manifest does not contain any 'monthly_shards'")

    # Support two manifest formats:
    # 1) A list of shard dicts (legacy)
    # 2) A dict mapping month_label -> shard_path (current generator)
    shards_list = []
    if isinstance(shards, dict):
        # Convert mapping into a list of shard-like dicts. We infer a 'processed'
        # flag from manifest['last_fine_tuned_month'] when available.
        last_ft = manifest.get("last_fine_tuned_month")
        for idx, (month_label, rel_path) in enumerate(sorted(shards.items())):
            shard = {
                "month": month_label,
                "shard_path": rel_path,
            }
            # mark processed if this month equals last_fine_tuned_month
            shard["processed"] = bool(last_ft == month_label)
            shards_list.append(shard)
    else:
        # Assume it's already a list of shard dicts
        shards_list = list(shards)

    unprocessed = []
    for idx, shard in enumerate(shards_list):
        if shard.get("processed", False):
            continue
        try:
            month_label, month_start, month_end = _infer_month_dates(shard)
        except ValueError:
            # Can't infer dates from this shard, skip it
            continue
        unprocessed.append((idx, shard, month_label, month_start, month_end))

    if not unprocessed:
        raise RuntimeError("No unprocessed monthly shards available for fine-tuning")

    # Pick the most recent month
    def _month_sort_key(item):
        _, _, month_label, _, _ = item
        return datetime.strptime(month_label, "%Y-%m")

    shard_idx, shard, month_label, month_start, month_end = max(unprocessed, key=_month_sort_key)

    base_dir = (
        shard.get("data_dir")
        or shard.get("base_dir")
        or manifest.get("base_dir")
        or f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    )

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Monthly shard data directory not found: {base_dir}")

    monthly_dataset = AllGraphDataSampler(
        base_dir=base_dir,
        date=True,
        train_start_date=month_start,
        train_end_date=month_end,
        mode="train",
    )

    if len(monthly_dataset) == 0:
        raise ValueError(f"Monthly dataset for {month_label} is empty (start={month_start}, end={month_end})")

    monthly_loader = DataLoader(
        monthly_dataset,
        batch_size=len(monthly_dataset),
        pin_memory=True,
        collate_fn=lambda x: x,
        drop_last=True,
    )

    env_init = create_env_init(args, data_loader=monthly_loader)

    checkpoint_candidates = [
        shard.get("checkpoint"),
        shard.get("checkpoint_path"),
        getattr(args, "resume_model_path", None),
        getattr(args, "baseline_checkpoint", None),
    ]
    checkpoint_candidates = [p for p in checkpoint_candidates if p]
    checkpoint_path = next((p for p in checkpoint_candidates if os.path.exists(p)), None)

    if checkpoint_path is None:
        raise FileNotFoundError("No valid base checkpoint found for fine-tuning")

    print(f"Fine-tuning {checkpoint_path} on month {month_label} ({month_start} to {month_end})")
    model = PPO.load(checkpoint_path, env=env_init, device=args.device)
    model.set_env(env_init)
    model.learn(total_timesteps=getattr(args, "fine_tune_steps", 5000))

    os.makedirs(args.save_dir, exist_ok=True)
    month_slug = month_label.replace("/", "-")
    out_path = os.path.join(args.save_dir, f"{args.model_name}_{month_slug}.zip")
    model.save(out_path)
    print(f"Saved fine-tuned checkpoint to {out_path}")

    # Update manifest bookkeeping
    shard.update({
        "processed": True,
        "checkpoint_path": out_path,
        "processed_at": datetime.utcnow().isoformat(timespec="seconds"),
    })
    manifest["monthly_shards"][shard_idx] = shard
    manifest["last_fine_tuned_month"] = month_label
    manifest["last_checkpoint_path"] = out_path

    output_manifest = bookkeeping_path or manifest_file
    with open(output_manifest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Updated manifest at {output_manifest}")

    return out_path

def train_predict(args, predict_dt):
    
    
    print(f"{'='*70}")
    print(f"PORTFOLIO CONSTRAINTS FROM RISK SCORE")
    print(f"{'='*70}")
    print(f"Risk Score (normalized): {args.risk_score:.3f}")
    print(f"Risk Description: {get_risk_profile_description(args.risk_score)}")
    print(f"\nPortfolio Constraints:")
    print(f"  Max Weight Cap: {args.risk_profile['max_weight']:.3f} ({args.risk_profile['max_weight']*100:.1f}% per stock)")
    print(f"  Min Weight Floor: {args.risk_profile['min_weight_floor']:.4f} ({args.risk_profile['min_weight_floor']*100:.2f}% minimum)")
    print(f"  Action Temperature: {args.risk_profile['action_temperature']:.3f}")
    print(f"  Target Positions: {args.risk_profile['target_num_positions']}")
    print(f"{'='*70}\n")
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


def load_user_risk_score(artifact_dir: str = './risk_artifacts', user_data: dict = None) -> tuple:
    """
    Load risk score from risk_cli artifacts with proper normalization.
    
    ACTUAL RANGES from artifacts:
    - Conservative: 0-20
    - Moderate: 35-65
    - Aggressive: 70-100
    
    Returns:
        (risk_score_normalized: float [0-1], metadata: dict)
        
    Raises:
        ValueError: If user_data is None (must be provided)
        FileNotFoundError: If artifacts not found
    """
    artifact_dir = os.path.abspath(artifact_dir)
    
    # ===== FAIL IF ARTIFACTS NOT FOUND =====
    if not os.path.isdir(artifact_dir):
        print(f"\n⚠ CRITICAL ERROR: Risk artifacts not found!")
        print(f"  Expected path: {artifact_dir}")
        print(f"  Solution: Run 'python risk_cli.py' first")
        raise FileNotFoundError(f"Risk artifacts not found at {artifact_dir}")
    
    # ===== FAIL IF NO USER DATA PROVIDED =====
    if user_data is None:
        print(f"\n⚠ CRITICAL ERROR: user_data is None!")
        print(f"  You MUST provide user risk data.")
        print(f"  Options:")
        print(f"    1. Create JSON file with user data: python main.py --user_risk_json user_profile.json")
        print(f"    2. Pass via code: modify main.py to provide user_data dict")
        raise ValueError("user_data must be provided - cannot use defaults")
    
    print(f"\n{'='*70}")
    print(f"LOADING RISK SCORE FROM ARTIFACTS (risk_cli)")
    print(f"{'='*70}")
    print(f"Artifact Directory: {artifact_dir}\n")
    
    # Load deployment pipeline
    scorer, offline_kmeans, cluster_context = load_deployment_pipeline(artifact_dir)
    
    print(f"Using CUSTOM user data provided")
    print(f"User Profile:")
    for key, value in user_data.items():
        print(f"  {key}: {value}")
    
    # Score the user with both offline and online methods
    print(f"\nScoring user with risk_cli...")
    offline_result = run_offline_inference(user_data, scorer, offline_kmeans, cluster_context)
    online_result = scorer.predict_and_update(user_data, update=False)
    
    # Use online result (includes drift detection)
    risk_score_raw = online_result['risk_score']  # This is 0-100
    risk_label = online_result['risk_label']
    cluster_id = online_result['cluster_id']
    
    # ===== CRITICAL: Normalize from [0-100] to [0-1] =====
    # risk_ranges from artifacts: Conservative: 0-20, Moderate: 35-65, Aggressive: 70-100
    risk_score_normalized = risk_score_raw / 100.0  # Convert 0-100 → 0-1
    risk_score_normalized = max(0.0, min(1.0, risk_score_normalized))  # Clamp to [0, 1]
    
    print(f"\n✓ RISK SCORE LOADED FROM ARTIFACTS")
    print(f"  Raw Score (0-100): {risk_score_raw:.2f}")
    print(f"  Normalized Score (0-1): {risk_score_normalized:.3f}")
    print(f"  Risk Label: {risk_label}")
    print(f"  Cluster ID: {cluster_id}")
    
    # Print optional fields if they exist
    if 'distance' in online_result:
        print(f"  Distance to Centroid: {online_result['distance']:.4f}")
    if 'reconstruction_error' in online_result:
        print(f"  Reconstruction Error: {online_result['reconstruction_error']:.4f}")
    if 'drift_detected' in online_result:
        print(f"  Drift Detected: {online_result['drift_detected']}")
    
    print(f"{'='*70}\n")
    
    metadata = {
        "risk_label": risk_label,
        "cluster_id": int(cluster_id) if cluster_id is not None else -1,
        "raw_score": float(risk_score_raw),
        "normalized_score": float(risk_score_normalized),
    }
    
    # Add optional fields if they exist
    if 'distance' in online_result:
        metadata['distance'] = float(online_result['distance'])
    if 'reconstruction_error' in online_result:
        metadata['reconstruction_error'] = float(online_result['reconstruction_error'])
    if 'drift_detected' in online_result:
        metadata['drift_detected'] = bool(online_result['drift_detected'])
    
    return risk_score_normalized, metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SmartFolio: Risk-Aware Portfolio Optimization")
    parser.add_argument("-device", "-d", default="cuda:0", help="gpu")
    parser.add_argument("-model_name", "-nm", default="SmartFolio", help="Model name used in checkpoints and logs")
    parser.add_argument("-horizon", "-hrz", default="1", help="Return prediction horizon in trading days")
    parser.add_argument("-relation_type", "-rt", default="hy", help="Correlation relation type label")
    parser.add_argument("-ind_yn", "-ind", default="y", help="Enable industry relation graph")
    parser.add_argument("-pos_yn", "-pos", default="y", help="Enable momentum relation graph")
    parser.add_argument("-neg_yn", "-neg", default="y", help="Enable reversal relation graph")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="Enable multi-reward IRL head")
    parser.add_argument("-policy", "-p", default="MLP", help="Policy architecture identifier")
    parser.add_argument("--artifact_dir", default="./risk_artifacts", help="Path to risk artifacts")
    parser.add_argument("--user_risk_json", required=True, help="[REQUIRED] Path to user risk data JSON file")
    parser.add_argument("--risk_score_override", type=float, default=None, help="Override risk score (0=conservative, 1=aggressive)")
    parser.add_argument("--resume_model_path", default=None, help="Path to previously saved PPO model to resume from")
    parser.add_argument("--reward_net_path", default=None, help="Path to saved IRL reward network state_dict to resume from")
    parser.add_argument("--fine_tune_steps", type=int, default=5000, help="Timesteps for monthly fine-tuning when resuming")
    parser.add_argument("--save_dir", default="./checkpoints", help="Directory to save trained models")
    parser.add_argument("--baseline_checkpoint", default="./checkpoints/baseline.zip",
                        help="Destination checkpoint promoted after passing gating criteria")
    parser.add_argument("--promotion_min_sharpe", type=float, default=0.5,
                        help="Minimum Sharpe ratio required to promote a fine-tuned checkpoint")
    parser.add_argument("--promotion_max_drawdown", type=float, default=0.2,
                        help="Maximum acceptable drawdown (absolute fraction, e.g. 0.2 for 20%) for promotion")
    parser.add_argument("--run_monthly_fine_tune", action="store_true",
                        help="Run monthly fine-tuning using the manifest instead of full training")
    parser.add_argument("--expert_cache_path", default=None,
                        help="Optional path to cache expert trajectories for reuse")
    parser.add_argument("--irl_epochs", type=int, default=50, help="Number of IRL training epochs")
    parser.add_argument("--rl_timesteps", type=int, default=10000, help="Number of RL timesteps for training")
    parser.add_argument("--dd_base_weight", type=float, default=1.0, help="Base weight for drawdown penalty")
    parser.add_argument("--dd_risk_factor", type=float, default=1.0, help="Risk factor k in β_dd(ρ) = β_base*(1+k*(1-ρ))")
    
    args = parser.parse_args()
    
    # ===== STEP 1: Load user risk data - REQUIRED =====
    if not args.user_risk_json or not os.path.exists(args.user_risk_json):
        print(f"\n⚠ CRITICAL ERROR: user_risk_json not found!")
        print(f"  Required argument: --user_risk_json /path/to/user_data.json")
        print(f"\nExample usage:")
        print(f"  python main.py \\")
        print(f"    --user_risk_json user_profile.json \\")
        print(f"    -policy HGAT \\")
        print(f"    --irl_epochs 5 \\")
        print(f"    --rl_timesteps 100")
        raise ValueError("--user_risk_json is REQUIRED")
    
    with open(args.user_risk_json, 'r') as f:
        args.user_risk_data = json.load(f)
    print(f"\n[✓] Loaded user risk data from {args.user_risk_json}")
    
    # ===== STEP 2: Set default values =====
    args.market = 'custom'
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.model_name = 'SmartFolio'
    args.relation_type = 'hy'
    args.train_start_date = '2020-01-06'
    args.train_end_date = '2023-01-31'
    args.val_start_date = '2023-02-01'
    args.val_end_date = '2023-12-29'
    args.test_start_date = '2024-01-02'
    args.test_end_date = '2024-12-26'
    args.batch_size = 32
    args.max_epochs = 1
    args.seed = 123
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True
    
    # ===== STEP 3: Auto-detect input_dim =====
    try:
        data_dir_detect = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        sample_files_detect = [f for f in os.listdir(data_dir_detect) if f.endswith('.pkl')]
        if sample_files_detect:
            import pickle
            sample_path_detect = os.path.join(data_dir_detect, sample_files_detect[50])
            with open(sample_path_detect, 'rb') as f:
                sample_data_detect = pickle.load(f)
            feats = sample_data_detect.get('features')
            if feats is not None:
                try:
                    shape = feats.shape
                except Exception:
                    try:
                        shape = feats.size()
                    except Exception:
                        shape = None
                if shape and len(shape) >= 2:
                    args.input_dim = shape[-1]
                    print(f"[✓] Auto-detected input_dim: {args.input_dim}")
                else:
                    args.input_dim = 6
            else:
                args.input_dim = 6
        else:
            args.input_dim = 6
    except Exception as e:
        print(f"[!] Warning: input_dim auto-detection failed ({e}); using 6")
        args.input_dim = 6
    
    # ===== STEP 4: Auto-detect num_stocks =====
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if sample_files:
        import pickle
        sample_path = os.path.join(data_dir, sample_files[50])
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        args.num_stocks = sample_data['features'].shape[0]
        print(f"[✓] Auto-detected num_stocks: {args.num_stocks}")
    else:
        raise ValueError(f"No pickle files found in {data_dir} to determine num_stocks")
    
    print(f"Market: {args.market}, Num Stocks: {args.num_stocks}")
    
    # ===== STEP 5: LOAD RISK SCORE FROM ARTIFACTS WITH USER DATA =====
    user_risk_data = args.user_risk_data
    risk_score_normalized, risk_metadata = load_user_risk_score(
        artifact_dir=getattr(args, 'artifact_dir', './risk_artifacts'),
        user_data=user_risk_data
    )
    
    args.risk_score = risk_score_normalized
    args.risk_metadata = risk_metadata
    
    # ===== STEP 6: BUILD RISK PROFILE FROM LOADED SCORE =====
    print("\n" + "="*70)
    print("BUILDING RISK PROFILE FROM LOADED SCORE")
    print("="*70)
    args.risk_profile = build_risk_profile(args.risk_score)
    
    print(f"\nRisk Profile Summary:")
    print(f"  Risk Score: {args.risk_profile['risk_score']:.3f}")
    print(f"  Max Weight: {args.risk_profile['max_weight']:.3f} ({args.risk_profile['max_weight']*100:.1f}%)")
    print(f"  Min Weight Floor: {args.risk_profile['min_weight_floor']:.4f} ({args.risk_profile['min_weight_floor']*100:.2f}%)")
    print(f"  Action Temperature: {args.risk_profile['action_temperature']:.3f}")
    print(f"  Target Positions: {args.risk_profile['target_num_positions']}")
    print(f"{'='*70}\n")
    
    # ===== STEP 7: Set remaining defaults =====
    args.irl_epochs = getattr(args, 'irl_epochs', 50)
    args.rl_timesteps = getattr(args, 'rl_timesteps', 10000)
    args.dd_base_weight = getattr(args, 'dd_base_weight', 1.0)
    args.dd_risk_factor = getattr(args, 'dd_risk_factor', 1.0)
    
    if not getattr(args, "expert_cache_path", None):
        args.expert_cache_path = os.path.join("dataset_default", "expert_cache")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ===== STEP 8: Run training =====
    if args.run_monthly_fine_tune:
        checkpoint = fine_tune_month(args, manifest_path="dataset_default/data_train_predict_custom/1_corr/monthly_manifest.json")
        print(f"\n[✓] Monthly fine-tuning complete. Checkpoint: {checkpoint}")
    else:
        trained_model = train_predict(args, predict_dt='2024-12-30')
        try:
            ts = time.strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(args.save_dir, f"ppo_{args.policy.lower()}_{args.market}_{ts}")
            print(f"\n[✓] Training run complete.")
            print(f"    Model checkpoint ready at: {out_path}")
        except Exception as e:
            print(f"[!] Note: {e}")







