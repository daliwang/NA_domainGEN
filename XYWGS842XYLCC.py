import pandas as pd
from pyproj import Transformer
import numpy as np
import csv

# Read the csv file
#AOI_gridcell_file = 'FluxNetNA.csv'
#AOI_gridcell_file = 'FluxNA_xcyc.csv'
AOI_gridcell_file = 'MOFLUX_xcyc.csv'
AOI = AOI_gridcell_file.split('_')[0]

#df = pd.read_csv(AOI_gridcell_file, sep=",", skiprows=1, names = ['site', 'yc', 'xc'], engine='python')
df = pd.read_csv(AOI_gridcell_file, sep=",", skiprows=1, names = ['xc', 'yc'], engine='python')

# Extract the 'yc (lat)' and 'xc (long)' columns
yc = df['yc'].astype(float)
xc = df['xc'].astype(float)

proj_daymet = "+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs" #custom CRS
transformer = Transformer.from_crs("EPSG:4326", proj_daymet)

# (lat, lon) -> y,x 
all_points = np.column_stack((yc,xc))
print(all_points)

xy_list=[]    #   transform(lat,long) -> x, y format in LCC
for pt in all_points:
    print(pt[0],pt[1])
    xy_list.append(transformer.transform(pt[0], pt[1]))

print(xy_list)

# Write to csv
output_csv_file = AOI+'_xcyc_LCC.csv'
with open(output_csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['xc_LCC', 'yc_LCC'])  # writing the header
    writer.writerows(xy_list)  # writing the data
