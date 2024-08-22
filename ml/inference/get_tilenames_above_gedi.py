import json

# Path to the GeoJSON file
geojson_file = '/scratch2/biomass_estimation/code/notebook/S2_tiles_Siberia_polybox/S2_tiles_Siberia_above_GEDI.geojson'

# Path to the output text file
output_file = '/scratch2/biomass_estimation/code/ml/inference/tile_names_inference.txt'

# Open the GeoJSON file
with open(geojson_file) as file:
    data = json.load(file)

# Extract the "Names" column
names = [feature['properties']['Name'] for feature in data['features']]

# Write the names to the output file
with open(output_file, 'w') as file:
    for name in names:
        file.write(name + '\n')