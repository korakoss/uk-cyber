import pandas as pd

COST_COLNAMES = [
    'Cybercrime_all',
    'ranscost_bands',      
    'hackcost_bands',      
    'tkvrcost_bands',      
    'doscost_bands',       
    'viruscost_bands',     
    'fraudcost_bands',     
    'fraudcost_bands2',    
    
    'hacksumcost_bands',   
    'notfraudcost_bands',  
    'crimecost_bands',     
    
    'damagedirsx_bands',   
    'damagedirlx_bands',   
    'damagestaffx_bands',  
    'damageindx_bands',    
    'damage_bands'         
]

TYPE_COLNAMES = [f'type{i}' for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16]]

# disrupta: type of most disruptive attack (all attacked firms, Q64A)
# disrupt:  same but only for firms with >1 breach
DISRUPT_COLNAMES = ['disrupta', 'disrupt']

# Per-attack count bands for phishing (Q89): the only attack type with
# high-coverage real count data. phishcon_bands = # targeted (personalised)
# phishing attacks; phisheng_bands = # attacks someone engaged with.
# Bands: 1=None, 2=1, 3=2-3, 4=4-5, 5=6-10, 6=11-20, 7=21-50, 8=51-100, 9=100+
PHISH_COUNT_COLNAMES = ['phishcon_bands', 'phisheng_bands']

# Impersonation severity proxies (Q53B/Q53C): whether impersonation instances
# involved a more severe secondary action. Ordinal: 1=Yes all of them,
# 2=Yes some of them, 3=No. fraud4_comb1/2 (Q88A) is a count/flag of breaches
# (any type) that resulted in impersonation as a downstream consequence —
# a related but distinct concept (consequence-of-any-breach, not
# severity-of-impersonation-itself).
IMPERSONATION_COLNAMES = ['impersonationhack', 'impersonationtkvr', 'fraud4_comb1', 'fraud4_comb2']

# Ransomware-specific detail (Q83 series) — a richer, previously-unloaded
# family analogous to phishing's phishcon_bands/phisheng_bands: a genuine
# per-attack COUNT variable (ranssoft_bands), plus the ransom DEMANDED and
# PAID amounts separately from the org's total reported cost (ranscost_bands).
# ranssoft_bands: count bands, 1=None,2=1,3=2-3,4=4-5,5=6-10,6=11-20,7=21-50,
# 8=51-100,9=100+ (same style as PHISH_COUNT_COLNAMES).
# ransdem_bands / ranspay_bands: 12-band £ scale, same as ranscost_bands.
# ranspayyn: 1=Yes totally, 2=Yes partially, 3=No.
RANSOMWARE_DETAIL_COLNAMES = ['ranssoft_bands', 'ransdem_bands', 'ranspay_bands', 'ranspayyn', 'ranschk']

# Fraud consequence counts (Q88A) — fraud1-3 are general fraud outcomes of
# ANY breach (money moved out / card misuse / fraudulent-invoice payment),
# not tagged by which attack type caused them, but plausibly relevant to
# Impersonation given its fraud-adjacent nature (fraud3 in particular reads
# like a classic invoice/BEC impersonation scam). fraud4 (already in
# IMPERSONATION_COLNAMES) is the one instance the survey DOES explicitly tie
# to impersonation-as-consequence. _comb1 = count (capped "5 or more"),
# _comb2 = any-incidents flag (0/1).
FRAUD_DETAIL_COLNAMES = [
    'fraud1_comb1', 'fraud1_comb2', 'fraud2_comb1', 'fraud2_comb2',
    'fraud3_comb1', 'fraud3_comb2',
]

# Cyber preparedness / security control variables (Q31 rules1-20, Q32
# policy1-13, Q33 series). rules1-20 and `trained` have full coverage (no
# routing/skip pattern); policy_comb, review, strategy, corprisk, update,
# insurex have substantial missingness from survey routing (e.g. only asked
# if firm has a policy at all) - check coverage before using.
RULES_COLNAMES = [f'rules{i}' for i in [1,2,3,4,5,7,8,9,10,11,13,14,15,17,18,19,20]]
PREPAREDNESS_COLNAMES = RULES_COLNAMES + [
    'rules_comb1', 'rules_comb2',
    'policy1','policy2','policy3','policy4','policy5','policy8','policy9','policy10','policy12','policy13','policy_comb',
    'review', 'trained', 'strategy', 'stratint', 'corporate', 'corprisk',
    'supplyrisk1', 'supplyrisk2', 'supplycert1', 'supplycert2', 'supplycert3',
    'update', 'insurex',
]

SPECIAL_CODE_THRESHOLD = 100  # damage_bands >= 100 are special codes (997, 999, 999.62...) — use < 100 to filter


def get_business_data(sfile: str = "data/proc/data.csv"):
    all_data = pd.read_csv(sfile)
    business_data = all_data[all_data["typex"] == 1]
    cols = ["sizeb", "weight", "freq"] + COST_COLNAMES + TYPE_COLNAMES + DISRUPT_COLNAMES + PHISH_COUNT_COLNAMES + IMPERSONATION_COLNAMES + RANSOMWARE_DETAIL_COLNAMES + FRAUD_DETAIL_COLNAMES + PREPAREDNESS_COLNAMES
    cols = [c for c in cols if c in all_data.columns]
    business_data = business_data[cols]
    return business_data


def get_charity_data(sfile: str = "data/proc/data.csv"):
    all_data = pd.read_csv(sfile)
    charity_data = all_data[all_data["typex"] == 2]
    cols = ["income2", "weight", "freq"] + COST_COLNAMES + TYPE_COLNAMES
    cols = [c for c in cols if c in all_data.columns]
    charity_data = charity_data[cols]
    return charity_data
