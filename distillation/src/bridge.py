"""Per-type bridge multipliers (shared constants).

The survey cost is the single MOST DISRUPTIVE incident; the bridge converts it to
a firm's TOTAL annual cost. The natural i.i.d. order-statistics bridge was
rejected (freq-cost is flat-to-negative, not increasing), so the bridge is a set
of per-attack-type multipliers, each derived in its own upstream analysis. These
enter the distillation as documented CONSTANTS with provenance, not re-derived.

Rules:
  - freq == 1 ("once only") -> bridge = 1.0 exactly (the reported cost IS the
    annual total; no bridging ambiguity). This overrides the type multiplier.
  - Impersonation (disrupta == 5) is a RANGE, not a point: leave-one-out on its
    small reference sample moves the estimate ~50%. Carried as low/high.
  - unresolved / minor types fall back to 1.0.

Provenance: src/estimation/bridge_specification.py (multipliers) and the per-type
derivations behind them (mixture_bridge.py for phishing; type_cost_censored_model
/ type_specific_bridge for malware/hacking/DoS/takeover; ransomware & impersonation
Monte-Carlo in ransomware_impersonation_rawdata.py / type_specific_montecarlo_bridge).
"""

# disrupta code -> multiplier
TYPE_BRIDGE = {
    1: 2.21,   # Ransomware
    2: 1.93,   # Other malware
    3: 0.36,   # Denial of service (< 1: extra incidents cheaper than the worst)
    4: 1.01,   # Hacking (broad)
    7: 1.01,   # Hacking (unauthorised access)
    9: 1.01,   # Hacking (other)
    6: 1.02,   # Phishing (mixture-model validated)
    11: 5.50,  # Website / social / email takeover
}

# Impersonation (disrupta == 5): a range, applied per replicate / reported low-high.
IMPERSONATION_LOW = 3.11
IMPERSONATION_HIGH = 6.25
IMPERSONATION_MID = (IMPERSONATION_LOW + IMPERSONATION_HIGH) / 2

# freq-based flat bridge (freq >= 2), the project's representative lower variant.
FREQ_BRIDGE = 1.02


def bridge(disrupta, freq, impersonation=IMPERSONATION_MID):
    """Type-based bridge multiplier for one firm.

    freq == 1 -> 1.0 (anchor). Impersonation -> the supplied `impersonation`
    value (default midpoint). Other/unknown types -> their multiplier or 1.0.
    """
    if freq == 1:
        return 1.0
    if disrupta is None:
        return 1.0
    try:
        d = int(disrupta)
    except (TypeError, ValueError):
        return 1.0
    if d == 5:
        return impersonation
    return TYPE_BRIDGE.get(d, 1.0)
