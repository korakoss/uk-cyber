# Notes on the UK dataset

## Relevant columns

### Respondent type
Raw column`typex`; (mildly) aggregated col `typex_comb`. Note: the survey also includes education institutions. Never _NaN_.


### Metadata per orgtype

#### Businesses
Here, we have a banded `sizeb` column for business size. This is never _NaN_ on businesses (`typex == 1`) and always _NaN_ otherwise. 

We also have the statistical adjustment coeff `weight`.

We also have some additional quantitive columns that could help in a deeper dive: data about recovery time after incidents, the economic sector of the business, and various measures of cyber preparedness.


#### Charities
These are `typex == 2`. The size information here is the `income2` column (again, non-NaN on all charities, always NaN otherwise). 

Beyond that, we have similar information as with businesses.


#### Institutions
This breaks down to several `typex` buckets per education level, but no size metadata. However, they do not seem to have cost or frequency data, so they should probably be scrubbed.


### Quantitative data structure (for businesses and charities)
There are several banded cost columns: columns per incident type, some aggregates, and also breakdowns of cost by cost type (defense, direct cost, outage).
