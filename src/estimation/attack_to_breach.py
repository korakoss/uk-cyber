"""How do attacks lead to breaches? Plain cross-tabs first (2026-10-08).
All attacked firms with a cost answer (n = 982). Breach marker as usual. Shares weighted, n unweighted.
A. Breached share by attack frequency, overall and by size.
B. Breached share by number of attack types ticked, and by frequency within number of types.
C. Within one type: breached share by that type's attack count (phishing count; firms whose only type(s) are
   phishing / impersonation, so the breach most likely came through them).
D. Breached share among firms that ticked exactly one type, by type.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7)) & df['damage_bands'].isin(range(1, 14))].copy()
num = lambda c: pd.to_numeric(df[c], errors='coerce')
df['breach'] = ((num('outcome_any') == 1) | num('restore').isin([3, 4, 5, 6]) |
                (num('impact1') == 1) | (num('impact2') == 1) | (num('impact4') == 1)).astype(int)
TYPES = {'type6': 'phishing', 'type5': 'impersonation', 'type1': 'ransomware', 'type2': 'malware', 'type3': 'DoS',
         'type4': 'bank hacking', 'type16': 'takeover', 'type8': 'outsider access', 'type7': 'staff access',
         'type15': 'eavesdropping', 'type9': 'other'}
df['k'] = (df[list(TYPES)] == 1).sum(axis=1)
FQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}


def share(s):
    return f'{np.average(s.breach, weights=s.weight):.2f} ({len(s)})' if len(s) else '-'


print('=== A. Breached share by attack frequency (n in brackets) ===')
print(f"{'':12s} {'all':>12s} {'Micro':>12s} {'Small':>12s} {'Medium+Large':>14s}")
for f, lab in FQ.items():
    s = df[df.freq == f]
    print(f'{lab:12s} {share(s):>12s} {share(s[s.sizeb == 1]):>12s} {share(s[s.sizeb == 2]):>12s} '
          f'{share(s[s.sizeb >= 3]):>14s}')
print(f"{'all':12s} {share(df):>12s} {share(df[df.sizeb == 1]):>12s} {share(df[df.sizeb == 2]):>12s} "
      f"{share(df[df.sizeb >= 3]):>14s}")

print('\n=== B. Breached share by number of types ticked, and by frequency within it ===')
print(f"{'types':8s} {'all':>12s} {'once':>12s} {'<mon-mon':>12s} {'weekly+':>12s}")
for k in [1, 2, 3, 4, 5]:
    s = df[(df.k == k) if k < 5 else (df.k >= 5)]
    print(f'{str(k) + ("+" if k == 5 else ""):8s} {share(s):>12s} {share(s[s.freq == 1]):>12s} '
          f'{share(s[s.freq.isin([2, 3])]):>12s} {share(s[s.freq >= 4]):>12s}')

print('\n=== C. Phishing count vs breached share (firms whose types are only phishing and/or impersonation) ===')
ps = num('Cybercrime_phishsum')
only_pi = (df.k >= 1) & ((df[['type6', 'type5']] == 1).sum(axis=1) == df.k) & (df.type6 == 1)
for lab, m in [('no count given', ~(ps >= 0)), ('1', ps == 1), ('2-5', ps.between(2, 5)), ('6-20', ps.between(6, 20)),
               ('21-100', ps.between(21, 100)), ('>100', ps > 100)]:
    print(f'   phishing count {lab:15s} {share(df[only_pi & m]):>12s}')
pcb = num('phishcon_bands')
print('   by targeted-phishing count (phishcon_bands): ' + ', '.join(
    f'{lab} {share(df[only_pi & m])}' for lab, m in
    [('none', pcb == 1), ('1-5', pcb.isin([2, 3, 4])), ('6-20', pcb.isin([5, 6])), ('21+', pcb.isin([7, 8, 9]))]))

print('\n=== D. Firms that ticked exactly one type: breached share by type ===')
for c, lab in TYPES.items():
    print(f'   {lab:16s} {share(df[(df.k == 1) & (df[c] == 1)]):>12s}')

# --- 2026-10-08: does exposure raise breach chances beyond which types reach the firm?
# Fit per-type breach chances p_t on all firms: P(breached) = 1 - prod over ticked types (1 - p_t) (types
# independent, no exposure term), with a size factor on the odds. Then compare observed vs predicted breached share
# across exposure measures. If heavily attacked firms breach more than their type mix implies, observed > predicted
# rises with exposure.
from scipy.optimize import minimize
from scipy.special import expit
X = (df[list(TYPES)] == 1).values.astype(float)
Zs = df.sizeb.values - 1
y = df.breach.values
w = (df.weight / df.weight.mean()).values


def pred(th):
    p = expit(th[:len(TYPES)][None, :] + th[-1] * Zs[:, None])
    return 1 - np.prod(np.where(X == 1, 1 - p, 1.0), axis=1)


def nll(th):
    q = np.clip(pred(th), 1e-9, 1 - 1e-9)
    return -(w * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum()


r = minimize(nll, np.r_[np.full(len(TYPES), -1.5), 0.1], method='L-BFGS-B')
df['pred'] = pred(r.x)
print('\n=== E. Per-type breach chances (Micro; odds x{:.2f} per size step), no exposure term ==='.format(np.exp(r.x[-1])))
print('   ' + ', '.join(f'{lab} {expit(v):.2f}' for lab, v in zip(TYPES.values(), r.x)))


def ovp(lab, s):
    if len(s) == 0:
        return
    print(f'   {lab:22s} n={len(s):3d}  observed {np.average(s.breach, weights=s.weight):.2f}  '
          f'predicted {np.average(s.pred, weights=s.weight):.2f}')


print('observed vs predicted breached share, by exposure measure:')
for f, lab in FQ.items():
    ovp(f'freq {lab}', df[df.freq == f])
for lab, m in [('phishing count 1', ps == 1), ('phishing 2-5', ps.between(2, 5)), ('phishing 6-20', ps.between(6, 20)),
               ('phishing >20', ps > 20)]:
    ovp(lab, df[m])
for lab, m in [('targeted phish none', pcb == 1), ('targeted 1-5', pcb.isin([2, 3, 4])), ('targeted 6+', pcb >= 5)]:
    ovp(lab, df[m])
for lab, m in [('1 type', df.k == 1), ('2 types', df.k == 2), ('3 types', df.k == 3), ('4+ types', df.k >= 4)]:
    ovp(lab, df[m])
