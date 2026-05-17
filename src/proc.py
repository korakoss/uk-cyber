import pandas as pd

all_data = pd.read_csv("data/proc/data.csv")

business_data = all_data[all_data["typex"] == 1].copy()
charity_data  = all_data[all_data["typex"] == 2].copy()



cost_columns = [
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

# Total: 15 cost-related columns

business_data = business_data[["sizeb", "weight", "freq"] + cost_columns]
charity_data = charity_data[["income2", "weight", "freq"] + cost_columns]


print(business_data)
print(charity_data)
