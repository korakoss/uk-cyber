# Findings

## Preword
This is based on the output of the files:
- basic_stats.py
- adv_b_stats.py

At present, I wrote up findings regarding businesses, as those are the input of the GovAI paper. But my impression is that charities generally give analogous results, so those might be added as "support for the case" later.

## Business findings

The incidence of experiencing any cybercrime shows a very clear monotonically increasing trend in business size, climbing from 18% in the smallest businneses to 52% in the largest ones.

Among the attacked businesses, frequency doesn't seem to strikingly depend on size -- there are some differences in the distributions, but nothing particularly systematic.

In terms of the reported (largest) damage and business size, we do seem to find a systematic trend of upwards movement. In particular, the costless attack category is leaking probability mass to higher bands (41%->27% between the two size extremes),


We can also investigate (cf. the _advstats_ file) the relationship between frequency and damage after conditioning on a given size band. We obvserve an interesting behavior: for the smaller bands (1 and 2), there is negative correlation, for the larger ones, positive. 

The direction of these correlations are consistent across various angles: Pearson and Spearman correlations agree, chi^2 confirms they are not independent, Cramér's V shows medium-size interdependence. 

Looking at the detailed tables, this behavior seems to arise for multiple reasons. First off, in lower bands, we generally see a sort of bifurcation between higher freq and higher damage; while in higher sizes, the "bottom right" region of the matrix is less sparse. At the same time, as we go towars larger sizes, the two variables also shift a bit upwards, which also contributes.
