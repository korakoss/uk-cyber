import pandas as pd

COST_COLNAMES = [
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

def get_business_data(sfile: str = "data/proc/data.csv"):
    all_data = pd.read_csv(sfile)
    business_data = all_data[all_data["typex"] == 1]
    business_data = business_data[["sizeb", "weight", "freq"] + COST_COLNAMES]
    return business_data


def get_charity_data(sfile: str = "data/proc/data.csv"):
    all_data = pd.read_csv(sfile)
    charity_data = all_data[all_data["typex"] == 1]
    charity_data = charity_data[["income2", "weight", "freq"] + COST_COLNAMES]
    return charity_data
