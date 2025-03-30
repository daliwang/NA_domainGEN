import pandas as pd
from pyproj import Transformer
import numpy as np
import csv
import sys

# Check if the input file is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python XYWGS842XYLCC.py <input_csv_file>")
    sys.exit(1)

# Read the input file from the command-line argument
AOI_gridcell_file = sys.argv[1]
AOI = AOI_gridcell_file.split('_')[0]  # Extract the prefix of the file name for output naming

# Read the CSV file into a DataFrame
# Assuming the input file has columns 'xc' (longitude) and 'yc' (latitude)
df = pd.read_csv(AOI_gridcell_file, sep=",", skiprows=1, names=['xc', 'yc'], engine='python')

# Extract the 'yc' (latitude) and 'xc' (longitude) columns as float arrays
yc = df['yc'].astype(float)
xc = df['xc'].astype(float)

# Define the Lambert Conformal Conic (LCC) projection for Daymet
proj_daymet = "+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
transformer = Transformer.from_crs("EPSG:4326", proj_daymet)  # Transformer for WGS84 to LCC

# Combine latitude and longitude into a single array for transformation
all_points = np.column_stack((yc, xc))

# Print the input points for debugging
print("Input points (latitude, longitude):")
print(all_points)

# Transform the points from WGS84 (lat, lon) to LCC (x, y)
xy_list = []  # List to store transformed points
for pt in all_points:
    print(f"Transforming point: lat={pt[0]}, lon={pt[1]}")
    xy_list.append(transformer.transform(pt[0], pt[1]))  # Transform and append to the list

# Print the transformed points for debugging
print("Transformed points (x, y in LCC):")
print(xy_list)

# Write the transformed points to a new CSV file
output_csv_file = AOI + '_xcyc_LCC.csv'
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['xc_LCC', 'yc_LCC'])  # Write the header
    writer.writerows(xy_list)  # Write the transformed data

print(f"Transformed coordinates saved to {output_csv_file}")