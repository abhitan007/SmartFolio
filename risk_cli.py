#!/usr/bin/env python3
"""CLI tool to score investor risk profiles using pre-trained artifacts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from river import drift


def _ensure_centers(model):
    if not hasattr(model, "centers"):
        model.centers = {}
    return model.centers


def kmeans_predict_manual(model, features):
    centers = _ensure_centers(model)
    if not centers:
        return None
    best_cid, best_dist = None, float("inf")
    keys = list(features.keys())
    for cid, center in centers.items():
        deltas = [features[k] - center.get(k, 0.0) for k in keys]
        dist = float(np.linalg.norm(deltas))
        if dist < best_dist:
            best_cid, best_dist = cid, dist
    return best_cid


def kmeans_learn_manual(model, features, lr=0.2):
    centers = _ensure_centers(model)
    cid = kmeans_predict_manual(model, features)
    if cid is None:
        new_id = len(centers)
        if new_id < model.n_clusters:
            centers[new_id] = features.copy()
        else:
            fallback = model.n_clusters - 1
            centers.setdefault(fallback, features.copy())
        return model
    center = centers[cid]
    for k, v in features.items():
        prev = center.get(k, 0.0)
        center[k] = prev + lr * (v - prev)
    centers[cid] = center
    model.centers = centers
    return model


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class OnlineRiskScorer:
    def __init__(
        self,
        autoencoder,
        river_kmeans,
        scaler,
        ohe,
        q_mappings,
        num_cols,
        cat_cols,
        feature_columns,
        risk_ranges,
        cluster_to_label,
        cluster_distance_stats,
        latent_dim,
    ):
        self.autoencoder = autoencoder
        self.kmeans = river_kmeans
        self.scaler = scaler
        self.ohe = ohe
        self.q_mappings = q_mappings
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.feature_columns = feature_columns
        self.risk_ranges = risk_ranges
        self.cluster_to_label = cluster_to_label
        self.cluster_distance_stats = cluster_distance_stats
        self.latent_dim = latent_dim
        self.update_cnt = 0
        self.drift_detector = drift.ADWIN()
        self.q_cols = list(q_mappings.keys())

    def _latent_to_dict(self, latent_vec):
        return {f"f{i}": float(val) for i, val in enumerate(latent_vec)}

    def _center_array(self, cluster_id):
        centers = _ensure_centers(self.kmeans)
        center_dict = centers.get(cluster_id, {})
        return np.array([center_dict.get(f"f{i}", 0.0) for i in range(self.latent_dim)])

    def _preprocess(self, sample_dict):
        df_sample = pd.DataFrame([sample_dict])
        for q in self.q_cols:
            df_sample[q] = df_sample[q].map(self.q_mappings[q])
        cat_transformed = self.ohe.transform(df_sample[self.cat_cols])
        df_cat = pd.DataFrame(cat_transformed, columns=self.ohe.get_feature_names_out(self.cat_cols))
        feature_cat_cols = [c for c in self.feature_columns if c not in self.num_cols and c not in self.q_cols]
        for col in feature_cat_cols:
            if col not in df_cat.columns:
                df_cat[col] = 0.0
        df_cat = df_cat.reindex(columns=feature_cat_cols, fill_value=0.0)
        df_num = pd.DataFrame(self.scaler.transform(df_sample[self.num_cols]), columns=self.num_cols)
        df_ord = df_sample[self.q_cols]
        df_processed = pd.concat([df_num, df_cat, df_ord], axis=1)
        df_processed = df_processed[self.feature_columns]
        return torch.tensor(df_processed.values, dtype=torch.float32)

    def predict_and_update(self, sample_dict, update=True):
        x_tensor = self._preprocess(sample_dict)
        self.autoencoder.eval()
        with torch.no_grad():
            latent_vec = self.autoencoder.encoder(x_tensor).cpu().numpy()[0]
            reconstruction = self.autoencoder(x_tensor)
        recon_error = float(torch.mean((reconstruction - x_tensor) ** 2).item())
        drift_alert = self.drift_detector.update(recon_error)
        latent_dict = self._latent_to_dict(latent_vec)
        cluster_id = kmeans_predict_manual(self.kmeans, latent_dict)
        if cluster_id is None:
            self.kmeans = kmeans_learn_manual(self.kmeans, latent_dict, lr=0.2)
            cluster_id = kmeans_predict_manual(self.kmeans, latent_dict)
        centroid_vec = self._center_array(cluster_id)
        dist = float(np.linalg.norm(latent_vec - centroid_vec))
        stats = self.cluster_distance_stats[cluster_id]
        stats["min"] = min(stats["min"], dist)
        stats["max"] = max(stats["max"], dist + 1e-6)
        denom = stats["max"] - stats["min"]
        dist_norm = (dist - stats["min"]) / denom if denom > 1e-8 else 0.0
        risk_label = self.cluster_to_label[cluster_id]
        r = self.risk_ranges[risk_label]
        risk_score = r["max"] - dist_norm * (r["max"] - r["min"])
        if update:
            self.kmeans = kmeans_learn_manual(self.kmeans, latent_dict, lr=0.2)
            self.update_cnt += 1
        return {
            "cluster_id": cluster_id,
            "risk_label": risk_label,
            "risk_score": float(risk_score),
            "distance": dist,
            "dist_norm": float(dist_norm),
            "reconstruction_error": recon_error,
            "drift_detected": bool(drift_alert),
        }

    @staticmethod
    def load(directory, autoencoder_class, input_dim, latent_dim):
        autoencoder = autoencoder_class(input_dim, latent_dim)
        autoencoder.load_state_dict(torch.load(os.path.join(directory, "autoencoder.pt"), map_location="cpu"))
        bundle = joblib.load(os.path.join(directory, "preprocessors.joblib"))
        kmeans = joblib.load(os.path.join(directory, "river_kmeans.joblib"))
        cluster_context = joblib.load(os.path.join(directory, "cluster_context.joblib"))
        return OnlineRiskScorer(
            autoencoder=autoencoder,
            river_kmeans=kmeans,
            scaler=bundle["scaler"],
            ohe=bundle["ohe"],
            q_mappings=bundle["q_mappings"],
            num_cols=bundle["num_cols"],
            cat_cols=bundle["cat_cols"],
            feature_columns=bundle["feature_columns"],
            risk_ranges=bundle["risk_ranges"],
            cluster_to_label=cluster_context["cluster_to_label"],
            cluster_distance_stats=cluster_context["cluster_distance_stats"],
            latent_dim=latent_dim,
        )


def load_deployment_pipeline(artifact_dir: str) -> Tuple[OnlineRiskScorer, Any, Dict[str, Any]]:
    bundle = joblib.load(os.path.join(artifact_dir, "preprocessors.joblib"))
    cluster_context = joblib.load(os.path.join(artifact_dir, "cluster_context.joblib"))
    input_dim = len(bundle["feature_columns"])
    latent_dim = cluster_context["latent_dim"]
    scorer = OnlineRiskScorer.load(
        directory=artifact_dir,
        autoencoder_class=Autoencoder,
        input_dim=input_dim,
        latent_dim=latent_dim,
    )
    offline_model = joblib.load(os.path.join(artifact_dir, "offline_kmeans.pkl"))
    return scorer, offline_model, cluster_context


def run_offline_inference(sample_dict: Dict[str, Any], scorer: OnlineRiskScorer, offline_model, cluster_context):
    x_tensor = scorer._preprocess(sample_dict)
    scorer.autoencoder.eval()
    with torch.no_grad():
        latent_vec = scorer.autoencoder.encoder(x_tensor).cpu().numpy()
    cluster_id = int(offline_model.predict(latent_vec)[0])
    centroid = np.array(cluster_context["centroids"][cluster_id])
    dist = float(np.linalg.norm(latent_vec - centroid))
    stats = cluster_context["cluster_distance_stats"][cluster_id]
    denom = stats["max"] - stats["min"]
    dist_norm = (dist - stats["min"]) / denom if denom > 1e-8 else 0.0
    risk_label = cluster_context["cluster_to_label"][cluster_id]
    r = cluster_context["risk_ranges"][risk_label]
    risk_score = r["max"] - dist_norm * (r["max"] - r["min"])
    return {
        "cluster_id": cluster_id,
        "risk_label": risk_label,
        "risk_score": float(risk_score),
        "distance": dist,
        "dist_norm": float(dist_norm),
    }


def parse_user_input(args) -> Dict[str, Any]:
    if args.user_json:
        with open(args.user_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("user_json must contain a JSON object with feature keys")
            return data
    return {
        "age": 70,
        "dependents": 2,
        "gross_income": 10_000_000,
        "tax_paid": 1_600_000,
        "gender": "Male",
        "main_occupation": "Salaried",
        "marital_status": "Married",
        "filing_timeliness": "On time",
        "Q1": "A",
        "Q2": "C",
        "Q3": "B",
        "Q4": "B",
        "Q5": "C",
        "Q6": "A",
    }


def format_report(offline_result, online_result):
    print("\n=== Offline (sklearn KMeans) ===")
    print(f"Risk Category : {offline_result['risk_label']}")
    print(f"Risk Score    : {offline_result['risk_score']:.2f}")
    print(f"Cluster ID    : {offline_result['cluster_id']}")
    print(f"Distance      : {offline_result['distance']:.4f} (norm {offline_result['dist_norm']:.4f})")

    print("\n=== Online (River KMeans) ===")
    print(f"Risk Category       : {online_result['risk_label']}")
    print(f"Risk Score          : {online_result['risk_score']:.2f}")
    print(f"Cluster ID          : {online_result['cluster_id']}")
    print(f"Distance            : {online_result['distance']:.4f} (norm {online_result['dist_norm']:.4f})")
    print(f"Reconstruction MSE  : {online_result['reconstruction_error']:.6f}")
    print(f"Drift detected      : {online_result['drift_detected']}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run investor risk scoring from saved artifacts.")
    parser.add_argument(
        "--artifact-dir",
        default=os.path.join(os.getcwd(), "risk_artifacts"),
        help="Path to the directory containing exported model artifacts.",
    )
    parser.add_argument(
        "--user-json",
        help="Path to a JSON file with the user input payload. If omitted, uses the sample from the notebook.",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Skip updating the online KMeans after scoring (pure inference mode).",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    artifact_dir = os.path.abspath(args.artifact_dir)
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    print(f"Loading artifacts from {artifact_dir}...")
    scorer, offline_kmeans, cluster_context = load_deployment_pipeline(artifact_dir)
    user_input = parse_user_input(args)

    print("Running offline scorer...")
    offline_result = run_offline_inference(user_input, scorer, offline_kmeans, cluster_context)

    print("Running online scorer...")
    online_result = scorer.predict_and_update(user_input, update=not args.no_update)

    format_report(offline_result, online_result)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
