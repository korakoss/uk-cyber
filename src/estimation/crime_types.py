"""Fresh look (2026-10-05) at crime types: how common each is, phishing's mass vs targeted split,
which types go with high attack frequency, and how costly each type is.
Businesses only, attacked firms (valid freq). Shares weighted; n unweighted.
'Targeted' phishing = phishcon_bands >= 2 (at least one phishing email containing the recipient's
personal details), which is the survey's own 'specifically targeted' measure.
"""
import numpy as np
import pandas as pd

df = pd.read_csv('data/proc/data.csv', low_memory=False)
df = df[(df['questtype'] == 1) & df['freq'].isin(range(1, 7))].copy()

TYPES = {'type6': 'phishing', 'type5': 'impersonation', 'type1': 'ransomware', 'type2': 'malware',
         'type3': 'DoS', 'type4': 'bank hacking', 'type16': 'takeover', 'type8': 'outsider access',
         'type7': 'staff access', 'type15': 'eavesdropping', 'type9': 'other'}
for c in TYPES:
    df[c] = (df[c] == 1).astype(int)
df['ntypes'] = df[list(TYPES)].sum(axis=1)

FREQ = {1: 'once', 2: '<monthly', 3: 'monthly', 4: 'weekly', 5: 'daily', 6: 'several/day'}
MID = {1: 0, 2: 50, 3: 300, 4: 750, 5: 3000, 6: 7500, 7: 15000, 8: 35000, 9: 75000,
       10: 300000, 11: 750000, 12: 3000000, 13: 7500000}
df['c'] = df['damage_bands'].where(df['damage_bands'].isin(MID.keys()))
df['cmid'] = df['c'].map(MID)


def w(sub):
    return sub['weight'].sum()


print('=== 0. Value coding check ===')
print({c: sorted(pd.read_csv('data/proc/data.csv', usecols=[c])[c].dropna().unique().tolist())[:6] for c in ['type6', 'type1']})

print('\n=== 1. How common is each type (attacked firms), and how often it is the only type ticked ===')
print(f'attacked firms n={len(df)}')
print(f"{'type':16s} {'n':>5s} {'share':>6s} {'only-type share of tickers':>28s}")
for c, name in TYPES.items():
    t = df[df[c] == 1]
    print(f'{name:16s} {len(t):5d} {w(t) / w(df):6.2f} {w(t[t["ntypes"] == 1]) / w(t):28.2f}')

ph = df[df['type6'] == 1].copy()
print('\n=== 2. Phishing count questions (among phishing tickers) ===')
BANDS = {1: 'none', 2: '1', 3: '2-3', 4: '4-5', 5: '6-10', 6: '11-20', 7: '21-50', 8: '51-100', 9: '100+'}
for col, label in [('phishcon_bands', 'targeted (personal details)'), ('phisheng_bands', 'engaged with')]:
    valid = ph[ph[col].isin(BANDS.keys())]
    print(f'-- {label}: answered {len(valid)} of {len(ph)} phishing tickers')
    print('   ' + '  '.join(f'{BANDS[b]}: {w(valid[valid[col] == b]) / w(valid):.2f}' for b in BANDS))

ph['targeted'] = np.where(ph['phishcon_bands'].isin(range(2, 10)), 'targeted',
                          np.where(ph['phishcon_bands'] == 1, 'mass only', 'unknown'))
ph['engaged'] = np.where(ph['phisheng_bands'].isin(range(2, 10)), 'engaged',
                         np.where(ph['phisheng_bands'] == 1, 'not engaged', 'unknown'))
print('\n-- targeted x engaged (weighted shares of phishing tickers, n in brackets)')
print(pd.crosstab(ph['targeted'], ph['engaged'], values=ph['weight'], aggfunc='sum', normalize='all').round(2))
print(pd.crosstab(ph['targeted'], ph['engaged']))

print('\n=== 3. Type mix by attack frequency ===')
print(f"{'freq':12s} {'n':>4s} {'phish':>6s} {'phish only':>10s} {'targeted':>9s} {'imp':>5s} {'mean #types':>11s}")
for f, lab in FREQ.items():
    s = df[df['freq'] == f]
    sp = ph[ph['freq'] == f]
    tshare = w(sp[sp['targeted'] == 'targeted']) / w(sp[sp['targeted'] != 'unknown']) if len(sp) else np.nan
    print(f'{lab:12s} {len(s):4d} {w(s[s.type6 == 1]) / w(s):6.2f} '
          f'{w(s[(s.type6 == 1) & (s.ntypes == 1)]) / w(s):10.2f} {tshare:9.2f} '
          f'{w(s[s.type5 == 1]) / w(s):5.2f} {np.average(s["ntypes"], weights=s["weight"]):11.2f}')
print('(targeted = share of phishing tickers with an answer who had >= 1 targeted email)')

print('\n=== 3b. Phishing counts by attack frequency (phishing tickers) ===')
print(f"{'freq':12s} {'n tgt':>5s} {'tgt>=6':>7s} {'tgt>=21':>8s} {'n eng':>6s} {'eng>=1':>7s} {'eng>=6':>7s}")
for f, lab in FREQ.items():
    sp = ph[ph['freq'] == f]
    a = sp[sp['phishcon_bands'].isin(BANDS.keys())]
    b = sp[sp['phisheng_bands'].isin(BANDS.keys())]
    print(f'{lab:12s} {len(a):5d} {w(a[a.phishcon_bands >= 5]) / w(a):7.2f} {w(a[a.phishcon_bands >= 7]) / w(a):8.2f} '
          f'{len(b):6d} {w(b[b.phisheng_bands >= 2]) / w(b):7.2f} {w(b[b.phisheng_bands >= 5]) / w(b):7.2f}')


def costrow(lab, s):
    s = s[s['c'].notna()]
    if len(s) == 0:
        return
    print(f'{lab:28s} {len(s):4d} {w(s[s.c == 1]) / w(s):6.2f} {w(s[s.c.isin([2, 3])]) / w(s):6.2f} '
          f'{w(s[s.c.isin([4, 5])]) / w(s):7.2f} {w(s[s.c >= 6]) / w(s):6.2f} '
          f'{np.average(s["cmid"], weights=s["weight"]):9,.0f}')


HEAD = f"{'':28s} {'n':>4s} {'none':>6s} {'<500':>6s} {'500-5k':>7s} {'5k+':>6s} {'mean £':>9s}"
print('\n=== 4a. Worst-incident cost by the TYPE of the worst incident (disrupta) ===')
print('(mean £ uses band midpoints; it is dominated by a few firms in the top bands)')
print(HEAD)
DIS = {6: 'phishing', 5: 'impersonation', 1: 'ransomware', 2: 'malware', 3: 'DoS', 4: 'bank hacking',
       11: 'takeover', 9: 'outsider access', 7: 'staff access', 10: 'eavesdropping', 12: 'other'}
dph = df.merge(ph[['imid', 'targeted']], on='imid', how='left')
for code, name in DIS.items():
    s = dph[dph['disrupta'] == code]
    costrow(name, s)
    if code == 6:
        costrow('  phishing, mass only', s[s['targeted'] == 'mass only'])
        costrow('  phishing, targeted', s[s['targeted'] == 'targeted'])
print('other disrupta codes:', dph.loc[~dph['disrupta'].isin(DIS.keys()), 'disrupta'].value_counts().to_dict())

print('\n=== 4b. Worst-incident cost for firms that ticked ONLY this type ===')
print(HEAD)
for c, name in TYPES.items():
    s = dph[(dph[c] == 1) & (dph['ntypes'] == 1)]
    costrow(name, s)
    if c == 'type6':
        costrow('  phishing only, mass only', s[s['targeted'] == 'mass only'])
        costrow('  phishing only, targeted', s[s['targeted'] == 'targeted'])

print('\n=== 5. Which firms drive each worst-type mean (share of weighted mean from the top firm) ===')
for code, name in DIS.items():
    s = dph[(dph['disrupta'] == code) & dph['c'].notna()]
    if len(s) < 5:
        continue
    contrib = s['cmid'] * s['weight']
    top = contrib.idxmax()
    print(f'{name:16s} top firm: band {int(s.loc[top, "c"])} size {int(s.loc[top, "sizeb"])} '
          f'-> {contrib[top] / contrib.sum():.2f} of the mean; mean without it '
          f'{(contrib.sum() - contrib[top]) / (s["weight"].sum() - s.loc[top, "weight"]):,.0f}')
