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
    cols = ["sizeb", "weight", "freq"] + COST_COLNAMES + TYPE_COLNAMES + DISRUPT_COLNAMES + PHISH_COUNT_COLNAMES + PREPAREDNESS_COLNAMES
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
