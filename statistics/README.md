# Statistical Analysis of Data

This folder includes code to analyze the distribution of ICESat and GEDI data and to compare one data source against the other.

## Files in this Repository
### `comparing_agb_gedi_agb.ipynb`
This Python notebook helps the user compare the aboveground biomass recorded in the ICESat mission against the values available through GEDI. 

###  `comparing_agb_latitude.ipynb`
This code compares the ICESat AGB to the latitude to check if there is a trend to over- or underestimate biomass in different latitudes.

## Other Notes
The first cell in these notebooks reads the GEDI shapefile which on our machine can take over 45 minutes. 

Also, there are csv files created for the different tiles to not have to repeat the loading procedures if the user requires a different analysis of the data. These interim results are saved in the comparisons_bins folder (empty in this github repo).