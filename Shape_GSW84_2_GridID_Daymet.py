import argparse
import geopandas as gpd
import xarray as xr
import numpy as np
import netCDF4 as nc
from pyproj import CRS, Transformer
from datetime import datetime

def shape2grid(shapefile_path, aoi_name, netcdf_file_path):
    # Get current date
    current_date = datetime.now()
    formatted_date = current_date.strftime('%y%m%d')

    # Load the shapefile
    shape = gpd.read_file(shapefile_path)
    print('Original Shapefile CRS:', shape.crs)

    # Define the Daymet CRS (Lambert Conformal Conic)
    daymet_crs = CRS.from_proj4("+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs")

    # Transform the shapefile to the Daymet CRS
    if shape.crs != daymet_crs:
        print(f"Converting shapefile CRS to Daymet CRS...")
        shape = shape.to_crs(daymet_crs.to_string())
    else:
        print(f"Shapefile is already in Daymet CRS.")

    # Load the NetCDF file
    ds = xr.open_dataset(netcdf_file_path)

    # Extract the xc and yc coordinates and the gridID
    xc = ds['xc'].values.squeeze()  # x-coordinates in Daymet CRS
    yc = ds['yc'].values.squeeze()  # y-coordinates in Daymet CRS
    gridID = ds['gridID'].values.squeeze()  # Grid IDs

    # Create a GeoDataFrame for the grid cells
    grid_cells = gpd.GeoDataFrame({
        'gridID': gridID,
        'geometry': gpd.points_from_xy(xc, yc)
    }, crs=daymet_crs)

    # Check which grid cells are within the AOI
    #grid_cells_within_AOI = grid_cells[grid_cells['geometry'].within(shape.union_all())]
    ''' 
    # Check which of the filtered grid cells are within the AOI
    print("Checking which grid cells are within the AOI...")
    grid_cells_within_AOI = []

    # Iterate over the filtered grid cells with a progress bar
    for _, row in tqdm(grid_cells_filtered.iterrows(), total=len(grid_cells_filtered), desc="Processing grid cells"):
        if row['geometry'].within(shape.geometry.union_all()):
            grid_cells_within_AOI.append(row)

    # Convert the result back to a GeoDataFrame
    grid_cells_within_AOI = gpd.GeoDataFrame(grid_cells_within_AOI, crs=daymet_crs)

    # Get the list of gridIDs that are inside the AOI
    grid_ids_within_AOI = grid_cells_within_AOI['gridID'].values
    '''

    # Filter grid cells by yc to reduce the number of candidates
    y_min, y_max = shape.bounds.miny.min(), shape.bounds.maxy.max()  # Get yc bounds of the shapefile
    grid_cells_filtered = grid_cells[(yc >= y_min) & (yc <= y_max)]

    # Check which of the filtered grid cells are within the AOI
    grid_cells_within_AOI = grid_cells_filtered[grid_cells_filtered['geometry'].within(shape.union_all())]


    # Get the list of gridIDs that are inside the AOI
    grid_ids_within_AOI = grid_cells_within_AOI['gridID'].values

    # Print the resulting grid IDs
    print(f"Grid IDs within {aoi_name}:", grid_ids_within_AOI)

    # Save the grid IDs to a NetCDF file
    AOI_gridID = f"{aoi_name}_gridID.c{formatted_date}.nc"
    dst = nc.Dataset(AOI_gridID, 'w', format='NETCDF3_64BIT')

    # Create dimensions
    ni_dim = dst.createDimension('ni', grid_ids_within_AOI.size)
    nj_dim = dst.createDimension('nj', 1)

    # Create variables
    gridID_var = dst.createVariable('gridID', np.int32, ('nj', 'ni'), zlib=True, complevel=5)
    gridID_var.long_name = f'Grid IDs within {aoi_name}'
    gridID_var.description = "Grid IDs of cells within the AOI in the Daymet domain"
    dst.variables['gridID'][...] = grid_ids_within_AOI

    dst.title = f'{aoi_name} land grid cells in the Daymet domain'
    dst.close()

    print(f"Grid IDs saved to {AOI_gridID}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find gridIDs within a shapefile AOI in the Daymet domain.')
    parser.add_argument('shapefile_path', type=str, help='Path to the shapefile')
    parser.add_argument('aoi_name', type=str, help='Name of the area of interest (AOI)')
    parser.add_argument('netcdf_file_path', type=str, help='Path to the Daymet NetCDF file')

    args = parser.parse_args()
    shape2grid(args.shapefile_path, args.aoi_name, args.netcdf_file_path)