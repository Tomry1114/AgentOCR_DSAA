from __future__ import annotations


def skill_signal_targets(
    *,
    clamp_signed_unit,
    skill_conf: float,
    reliability: float,
    support: float,
    rescue: float,
    utility: float,
    uncertainty: float = 0.0,
    role_utility: float = 0.0,
    witness_utility: float = 0.0,
    witness_mode: bool,
    adaptive_skill_mix: bool,
) -> dict[str, float]:
    skill_conf = min(1.0, max(0.0, float(skill_conf)))
    reliability = min(1.0, max(0.0, float(reliability)))
    support = min(1.0, max(0.0, float(support)))
    rescue = min(2.5, max(0.0, float(rescue)))
    rescue_norm = min(1.0, rescue / 2.5)
    utility = clamp_signed_unit(utility)
    uncertainty = min(1.0, max(0.0, float(uncertainty)))
    role_positive = min(1.0, max(0.0, float(role_utility)))
    witness_positive = min(1.0, max(0.0, float(witness_utility)))

    fixed_targets = (
        {
            "query_relevance": min(
                1.0,
                0.36
                + 0.16 * skill_conf
                + 0.10 * support
                + 0.10 * rescue
                + 0.10 * utility
                + 0.06 * uncertainty
                + 0.32 * role_utility
                + 0.22 * witness_utility,
            ),
            "salience": min(
                1.0,
                0.38
                + 0.18 * skill_conf
                + 0.08 * support
                + 0.10 * rescue
                + 0.08 * utility
                + 0.04 * uncertainty
                + 0.24 * role_utility
                + 0.18 * witness_utility,
            ),
            "source_trust": min(
                1.0,
                0.42
                + 0.16 * skill_conf
                + 0.08 * reliability
                + 0.08 * rescue
                + 0.08 * utility
                - 0.06 * uncertainty
                + 0.18 * role_utility
                + 0.20 * witness_utility,
            ),
            "support_bonus": support + rescue + utility + 0.75 * uncertainty + 1.5 * role_utility + witness_utility,
        }
        if witness_mode
        else {
            "query_relevance": min(1.0, 0.45 + 0.18 * reliability + 0.10 * support + 0.08 * rescue + 0.10 * utility - 0.06 * uncertainty),
            "salience": min(1.0, 0.48 + 0.15 * reliability + 0.08 * support + 0.08 * rescue + 0.08 * utility - 0.05 * uncertainty),
            "source_trust": min(1.0, 0.50 + 0.16 * reliability + 0.08 * support + 0.08 * rescue + 0.10 * utility - 0.10 * uncertainty),
            "support_bonus": support + rescue + utility,
        }
    )

    if not adaptive_skill_mix:
        return fixed_targets

    positive_utility = max(0.0, utility)
    negative_utility = max(0.0, -utility)
    evidence = min(1.0, (0.42 * reliability + 0.20 * support + 0.16 * rescue_norm + 0.22 * positive_utility) * (1.0 - 0.45 * uncertainty))
    instability = min(1.0, 0.35 * (1.0 - reliability) + 0.24 * negative_utility + 0.16 * (1.0 - support) + 0.25 * uncertainty)
    role_bonus = 0.18 * role_positive if witness_mode else 0.0
    witness_bonus = 0.14 * witness_positive if witness_mode else 0.0
    conf_bonus = 0.12 * skill_conf if witness_mode else 0.0

    query_target = min(
        1.0,
        0.44
        + 0.34 * evidence
        + conf_bonus
        + role_bonus
        + witness_bonus
        + (0.08 * uncertainty if witness_mode else -0.05 * uncertainty)
        - 0.06 * instability,
    )
    salience_target = min(
        1.0,
        0.46
        + 0.30 * evidence
        + (0.14 * skill_conf if witness_mode else 0.0)
        + 0.12 * role_positive
        + 0.10 * witness_positive
        + (0.06 * uncertainty if witness_mode else -0.04 * uncertainty)
        - 0.05 * instability,
    )
    trust_target = min(
        1.0,
        0.48
        + 0.32 * evidence
        + 0.10 * reliability
        + (0.08 * skill_conf if witness_mode else 0.0)
        + 0.08 * role_positive
        + 0.10 * witness_positive
        - 0.10 * uncertainty
        - 0.04 * instability,
    )
    support_bonus = support + rescue_norm + positive_utility + (0.70 * uncertainty if witness_mode else 0.0) + (1.2 * role_positive if witness_mode else 0.0) + (0.8 * witness_positive if witness_mode else 0.0)
    adaptive_targets = {
        "query_relevance": query_target,
        "salience": salience_target,
        "source_trust": trust_target,
        "support_bonus": support_bonus,
    }
    if not witness_mode:
        return adaptive_targets

    if role_utility < -0.05 or witness_utility < -0.05:
        return fixed_targets

    positive_evidence = min(1.0, 0.28 * skill_conf + 0.24 * reliability + 0.20 * support + 0.16 * rescue_norm + 0.12 * positive_utility)
    positive_credit = min(1.0, 0.65 * role_positive + 0.35 * witness_positive)
    if positive_credit <= 1e-8 and positive_evidence < 0.35:
        return fixed_targets

    boost = min(0.08, 0.03 * positive_evidence + 0.05 * positive_credit)
    return {
        "query_relevance": min(1.0, max(fixed_targets["query_relevance"], fixed_targets["query_relevance"] + boost)),
        "salience": min(1.0, max(fixed_targets["salience"], fixed_targets["salience"] + 0.80 * boost)),
        "source_trust": min(1.0, max(fixed_targets["source_trust"], fixed_targets["source_trust"] + 0.70 * boost)),
        "support_bonus": fixed_targets["support_bonus"] + 0.40 * positive_evidence + 0.60 * positive_credit,
    }
