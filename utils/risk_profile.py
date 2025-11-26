"""Utility helpers for mapping a user risk score into concrete knobs."""

from __future__ import annotations


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, float(value)))


def _lerp(low: float, high: float, weight: float) -> float:
    return low + (high - low) * weight


"""
Risk profile builder: converts normalized risk_score [0-1] to portfolio constraints.

ACTUAL ARTIFACT RANGES:
- Conservative: 0-20
- Moderate: 35-65
- Aggressive: 70-100

After normalization to [0-1]:
- Conservative: 0.0-0.2
- Moderate: 0.35-0.65
- Aggressive: 0.7-1.0
"""


def build_risk_profile(risk_score: float) -> dict:
    """
    Build portfolio constraints from normalized risk_score [0-1].
    
    Args:
        risk_score: Float in [0, 1]
                   0.0 = Conservative (capital preservation)
                   0.5 = Moderate (balanced)
                   1.0 = Aggressive (growth-focused)
    
    Returns:
        dict: Portfolio constraints derived from risk_score
    
    Mapping:
        risk_score=0.15 → max_weight=0.14, min_floor=0.006 (very diversified)
        risk_score=0.50 → max_weight=0.30, min_floor=0.0125 (balanced)
        risk_score=0.85 → max_weight=0.44, min_floor=0.018 (concentrated)
    """
    
    risk_score = max(0.0, min(1.0, float(risk_score)))
    
    profile = {
        'risk_score': risk_score,
        
        # MAX WEIGHT CAP: Linear interpolation 0.10 → 0.50
        # Conservative users: max 10% per stock (force diversification)
        # Aggressive users: max 50% per stock (allow concentration)
        'max_weight': 0.10 + (0.40 * risk_score),
        
        # MIN WEIGHT FLOOR: Linear interpolation 0.005 → 0.02
        # Conservative: 0.5% minimum (many positions)
        # Aggressive: 2% minimum (fewer positions, meaningful size)
        'min_weight_floor': 0.005 + (0.015 * risk_score),
        
        # TARGET NUMBER OF POSITIONS: Interpolate 30 → 10
        'target_num_positions': int(30 - (20 * risk_score)),
        
        # ACTION TEMPERATURE: Controls softmax spread
        # Higher temp → More uniform → More diversified
        # Lower temp → Sharper peaks → More concentrated
        'action_temperature': max(1e-3, 1.0 + (1.0 - risk_score)),
        
        # PENALTIES for reward shaping
        'variance_penalty': 1.0 - (0.5 * risk_score),        # Higher for conservative
        'concentration_penalty': 0.5 * (1.0 - risk_score),   # Higher for conservative
        'drawdown_penalty': 0.5 + (0.5 * risk_score),        # Higher for aggressive
    }
    
    # Ensemble weights for multi-objective learning
    profile['ensemble_weights'] = {
        'return_weight': 0.3 + (0.4 * risk_score),           # 0.3 → 0.7
        'risk_weight': 1.0 - (0.3 * risk_score),             # 1.0 → 0.7
        'diversification_weight': 1.0 - (0.3 * risk_score),  # 1.0 → 0.7
    }
    
    return profile


def get_risk_profile_description(risk_score: float) -> str:
    """Get human-readable description of risk profile."""
    risk_score = max(0.0, min(1.0, float(risk_score)))
    
    # Map to actual artifact ranges
    if risk_score < 0.20:
        return "Very Conservative (0-20: Capital Preservation, High Diversification)"
    elif risk_score < 0.35:
        return "Conservative (20-35: Low Risk, Diversified)"
    elif risk_score < 0.50:
        return "Moderate-Conservative (35-50: Balanced with Safety)"
    elif risk_score < 0.65:
        return "Moderate (50-65: Balanced Risk-Return)"
    elif risk_score < 0.80:
        return "Aggressive (65-80: Growth-Focused)"
    else:
        return "Very Aggressive (80-100: High Growth, Concentrated)"
