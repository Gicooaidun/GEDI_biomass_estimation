# icesat_data

This folder will be the location where the ICESat data will be downloaded to by a bash script. An example script is included here.

## Contents

After completing the data_preprocessing pipeline the folder should include many files of the format

- ...
- boreal_agb_[number].tif
- ...


These files don't contain the ICESat data. The ICESat shapefiles which show the limits of the ICESat tiles and other metadata are saved in the boreal_agb_density_ICESat2_tiles_shp folder.