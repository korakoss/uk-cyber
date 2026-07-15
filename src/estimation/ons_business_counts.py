"""
Step 1 of the estimation pipeline: N(size), the UK business population count
per employment size band. This is the external input that converts every
per-business figure produced elsewhere into an actual national total.

Source: ONS / DBT "Business population estimates for the UK and regions 2025"
(statistical release, start-of-2025 figures), GOV.UK:
https://www.gov.uk/government/statistics/business-population-estimates-2025/business-population-estimates-for-the-uk-and-regions-2025-statistical-release
Fetched 2026-07-15.

FRAME CHOICE — employer businesses only (1+ employees):
The Cyber Security Breaches Survey size bands (sizeb, per prevalence.py) are
Micro (1-9), Small (10-49), Medium (50-249), Large (250+) — i.e. the Micro
band starts at 1 employee, NOT 0. The ~4.27 million zero-employee businesses
(sole traders / no employees aside from owner) therefore fall OUTSIDE the
survey frame and are excluded here. We use the ONS employer breakdown, whose
bands line up exactly with the survey's sizeb bands.

If a future decision is to instead treat the whole private-sector population
(including the 0-employee businesses) as in-scope, swap MICRO to include
N_ZERO_EMPLOYEE below and revisit whether the Micro prevalence/cost figures
(estimated on 1-9 firms) can be extended to 0-employee firms — they probably
can't without an assumption, which is exactly why the employer frame is the
clean default.

Consistency check (start-of-2025): the four employer bands sum to the ONS
"employers" total of 1,417,730.
"""

# Start-of-2025 ONS figures, employer businesses (1+ employees), by size band.
# Keys match prevalence.py's sizeb codes: 1=Micro, 2=Small, 3=Medium, 4=Large.
N_BY_SIZE = {
    1: 1_150_875,  # Micro  (1-9 employees)
    2:   220_085,  # Small  (10-49)
    3:    38_435,  # Medium (50-249)
    4:     8_335,  # Large  (250+)
}

SIZE_LABELS = {1: 'Micro (1-9)', 2: 'Small (10-49)', 3: 'Medium (50-249)', 4: 'Large (250+)'}

# Context figures from the same release (not used for scaling employer-frame
# estimates, kept for reference / alternative-frame sensitivity).
N_ZERO_EMPLOYEE = 4_272_535
N_TOTAL_PRIVATE_SECTOR = 5_690_265
N_EMPLOYERS_TOTAL = 1_417_730

if __name__ == '__main__':
    print("ONS Business Population Estimates 2025 — employer businesses (1+ employees)")
    print("=" * 70)
    total = 0
    for sz in sorted(N_BY_SIZE):
        n = N_BY_SIZE[sz]
        total += n
        print(f"  {SIZE_LABELS[sz]:<18} {n:>12,}")
    print("-" * 70)
    print(f"  {'Sum of bands':<18} {total:>12,}")
    print(f"  {'ONS employers total':<18} {N_EMPLOYERS_TOTAL:>12,}")
    assert total == N_EMPLOYERS_TOTAL, "band sum does not match ONS employers total"
    print("  ✓ consistency check passed (bands sum to ONS employers total)")
    print()
    print(f"  (Excluded from frame: {N_ZERO_EMPLOYEE:,} zero-employee businesses;")
    print(f"   total private sector = {N_TOTAL_PRIVATE_SECTOR:,})")
