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
from utils.risk_profile import build_risk_profile

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
    parser.add_argument("-model_name", "-nm", default="SmartFolio", help="Model name used in checkpoints and logs")
    parser.add_argument("-horizon", "-hrz", default="1", help="Return prediction horizon in trading days")
    parser.add_argument("-relation_type", "-rt", default="hy", help="Correlation relation type label")
    parser.add_argument("-ind_yn", "-ind", default="y", help="Enable industry relation graph")
    parser.add_argument("-pos_yn", "-pos", default="y", help="Enable momentum relation graph")
    parser.add_argument("-neg_yn", "-neg", default="y", help="Enable reversal relation graph")
    parser.add_argument("-multi_reward_yn", "-mr", default="y", help="Enable multi-reward IRL head")
    parser.add_argument("-policy", "-p", default="MLP", help="Policy architecture identifier")
    # continual learning / resume
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
    # Training hyperparameters
    parser.add_argument("--irl_epochs", type=int, default=50, help="Number of IRL training epochs")
    parser.add_argument("--rl_timesteps", type=int, default=10000, help="Number of RL timesteps for training")
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Skip configuring TensorBoard logging to avoid importing the optional dependency.",
    )
    # Risk-adaptive reward parameters
    parser.add_argument("--risk_score", type=float, default=0.5, help="User risk score: 0=conservative, 1=aggressive")
    parser.add_argument("--dd_base_weight", type=float, default=1.0, help="Base weight for drawdown penalty")
    parser.add_argument("--dd_risk_factor", type=float, default=1.0, help="Risk factor k in β_dd(ρ) = β_base*(1+k*(1-ρ))")
    args = parser.parse_args()
    args.market = 'custom'

    if getattr(args, "disable_tensorboard", False):
        PPO_PARAMS["tensorboard_log"] = None
        print("TensorBoard logging disabled (--disable-tensorboard); PPO will not attempt to import tensorboard.")

    # Default training range (override via CLI if desired)
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
    # Auto-detect input_dim (number of per-stock features) from a sample file
    try:
        data_dir_detect = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
        sample_files_detect = [f for f in os.listdir(data_dir_detect) if f.endswith('.pkl')]
        if sample_files_detect:
            import pickle
            sample_path_detect = os.path.join(data_dir_detect, sample_files_detect[0])
            with open(sample_path_detect, 'rb') as f:
                sample_data_detect = pickle.load(f)
            # Expect features shaped [T, num_stocks, input_dim]
            feats = sample_data_detect.get('features')
            if feats is not None:
                # Handle both torch tensors and numpy arrays
                try:
                    shape = feats.shape
                except Exception:
                    # If it's a torch tensor wrapped differently
                    try:
                        shape = feats.size()
                    except Exception:
                        shape = None
                if shape and len(shape) >= 2:
                    args.input_dim = shape[-1]
                    print(f"Auto-detected input_dim: {args.input_dim}")
                else:
                    print("Warning: could not determine input_dim from sample; falling back to 6")
                    args.input_dim = 6
            else:
                print("Warning: 'features' not found in sample; falling back to input_dim=6")
                args.input_dim = 6
        else:
            print(f"Warning: No sample files found in {data_dir_detect}; falling back to input_dim=6")
            args.input_dim = 6
    except Exception as e:
        print(f"Warning: input_dim auto-detection failed ({e}); falling back to 6")
        args.input_dim = 6
    args.ind_yn = True
    args.pos_yn = True
    args.neg_yn = True
    args.multi_reward = True
    # Training hyperparameters (can be overridden via command line)
    args.irl_epochs = getattr(args, 'irl_epochs', 50)
    args.rl_timesteps = getattr(args, 'rl_timesteps', 10000)
    # Risk-adaptive reward parameters
    args.risk_score = getattr(args, 'risk_score', 0.5)
    args.dd_base_weight = getattr(args, 'dd_base_weight', 1.0)
    args.dd_risk_factor = getattr(args, 'dd_risk_factor', 1.0)
    args.risk_profile = build_risk_profile(args.risk_score)
    if not getattr(args, "expert_cache_path", None):
        args.expert_cache_path = os.path.join(
            "dataset_default",
            "expert_cache"
        )
    # ensure save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Auto-detect num_stocks from a sample pickle file
    data_dir = f'dataset_default/data_train_predict_{args.market}/{args.horizon}_{args.relation_type}/'
    sample_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    if sample_files:
        import pickle
        sample_path = os.path.join(data_dir, sample_files[0])
        with open(sample_path, 'rb') as f:
            sample_data = pickle.load(f)
        # features shape is [num_stocks, feature_dim], so use shape[0]
        args.num_stocks = sample_data['features'].shape[0]
        print(f"Auto-detected num_stocks for custom market: {args.num_stocks}")
    else:
        raise ValueError(f"No pickle files found in {data_dir} to determine num_stocks")
    print("market:", args.market, "num_stocks:", args.num_stocks)
    if args.run_monthly_fine_tune:
        checkpoint = fine_tune_month(args, manifest_path="dataset_default/data_train_predict_custom/1_corr/monthly_manifest.json")
        print(f"Monthly fine-tuning complete. Checkpoint: {checkpoint}")
    else:
        trained_model = train_predict(args, predict_dt='2024-12-30')
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
