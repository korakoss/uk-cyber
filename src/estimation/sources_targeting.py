"""Do the 'mass' (volume) source and the 'breachy' source target firms independently (H1), or prefer different
firms (H2)? (2026-10-06)
Per-firm shares among ALL attacked firms with a cost answer (no conditioning on the breach marker), by phishing
volume, within Micro / Small+ and pooled:
  - costly: worst incident >= £5k
  - costly compromise: costly AND an 'inside' type ticked (ransomware, bank hacking, outsider / staff access)
  - inside type ticked at all
  - breached (marker), for reference
H1 predicts the costly / costly-compromise shares are flat or rising with volume; H2 predicts they fall.
Volume measures (each has drawbacks):
  A. exact phishing count (Cybercrime_phishsum): best measure, but answered selectively (more by broad firms);
     firms not ticking phishing shown as their own row
  B. targeted phishing count (phishcon_bands): answered by most phishing firms
  C. overall attack frequency (freq): answered by all, but covers every type
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
df['inside'] = ((df[['type1', 'type4', 'type8', 'type7']] == 1).any(axis=1)).astype(int)
df['costly'] = (df['damage_bands'] >= 6).astype(int)
df['comp'] = df.costly * df.inside
df['phish'] = (df['type6'] == 1).astype(int)
ps = num('Cybercrime_phishsum')
pcb = num('phishcon_bands')

SIZES = [('all sizes', df.sizeb > 0), ('Micro', df.sizeb == 1), ('Small+', df.sizeb >= 2)]


def table(title, groups):
    print(f'\n=== {title} ===')
    print(f"{'':12s} {'band':16s} {'n':>4s} {'costly':>7s} {'costly+inside':>14s} {'inside':>7s} {'breached':>9s}")
    for slab, sm in SIZES:
        for glab, gm in groups:
            s = df[sm & gm]
            if len(s) < 5:
                continue
            w = s.weight
            print(f'{slab:12s} {glab:16s} {len(s):4d} {np.average(s.costly, weights=w):7.3f} '
                  f'{np.average(s.comp, weights=w):14.3f} {np.average(s.inside, weights=w):7.2f} '
                  f'{np.average(s.breach, weights=w):9.2f}')
        print()


table('A. Exact phishing count', [
    ('no phishing', df.phish == 0),
    ('phish, no count', (df.phish == 1) & ~(ps >= 0)),
    ('1', ps == 1), ('2-5', ps.between(2, 5)), ('6-20', ps.between(6, 20)),
    ('21-100', ps.between(21, 100)), ('>100', ps > 100)])
table('B. Targeted phishing count', [
    ('no phishing', df.phish == 0), ('none', pcb == 1), ('1-5', pcb.isin([2, 3, 4])),
    ('6-20', pcb.isin([5, 6])), ('21+', pcb.isin([7, 8, 9]))])
FREQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
table('C. Overall attack frequency', [(lab, df.freq == f) for f, lab in FREQ.items()])
