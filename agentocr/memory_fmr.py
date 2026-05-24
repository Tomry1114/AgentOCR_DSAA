from __future__ import annotations


def beta_uncertainty(alpha: float, beta: float) -> float:
    alpha = max(1e-6, float(alpha))
    beta = max(1e-6, float(beta))
    total = alpha + beta
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    variance_score = min(1.0, variance / (1.0 / 12.0))
    scarcity = min(1.0, 3.0 / (total + 1.0))
    return min(1.0, max(0.0, 0.65 * variance_score + 0.35 * scarcity))


def update_uncertainty_state(
    *,
    previous_alpha: float,
    previous_beta: float,
    success_signal: float,
    failure_signal: float,
    support_signal: float,
    conflict_signal: float,
    utility_signal: float,
    next_reliability: float,
    next_conflict: float,
) -> tuple[float, float, float]:
    alpha_signal = max(0.0, 0.75 * success_signal + 0.55 * support_signal + 0.30 * max(0.0, utility_signal))
    beta_signal = max(0.0, 0.85 * failure_signal + 0.55 * conflict_signal + 0.35 * max(0.0, -utility_signal))
    evidence_decay = min(0.98, max(0.80, 0.90 + 0.05 * next_reliability - 0.04 * next_conflict))
    next_alpha = float(previous_alpha) * evidence_decay + alpha_signal
    next_beta = float(previous_beta) * evidence_decay + beta_signal
    next_uncertainty = beta_uncertainty(1.0 + next_alpha, 1.0 + next_beta)
    return next_alpha, next_beta, next_uncertainty
