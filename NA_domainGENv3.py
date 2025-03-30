import os
import math
import netCDF4 as nc
import numpy as np
from pyproj import Transformer, CRS
from datetime import datetime
from time import process_time

def calculate_vertices(Txy2lonlat, XC, YC, x_offset, y_offset):
    """
    Calculate the four vertices of grid cells and return xv and yv arrays.

    Parameters:
        Txy2lonlat: Transformer object to convert coordinates.
        XC: Array of x-coordinates of grid cell centers.
        YC: Array of y-coordinates of grid cell centers.
        x_offset: Offset in the x-direction (half the grid cell width).
        y_offset: Offset in the y-direction (half the grid cell height).

    Returns:
        xv: NumPy array of x-coordinates of the four vertices.
        yv: NumPy array of y-coordinates of the four vertices.
    """
    x_offsets = [-x_offset, x_offset, x_offset, -x_offset]
    y_offsets = [y_offset, y_offset, -y_offset, -y_offset]

    xv, yv = [], []
    for dx, dy in zip(x_offsets, y_offsets):
        x, y = Txy2lonlat.transform(XC + dx, YC + dy)
        xv.append(x)
        yv.append(y)

    return np.array(xv), np.array(yv)

def create_projection_variable(nc_file, name, **kwargs):
    """
    Helper function to create a projection variable in a NetCDF file.

    Parameters:
        nc_file: NetCDF file object.
        name: Name of the projection variable.
        kwargs: Attributes for the projection variable.
    """
    var = nc_file.createVariable(name, np.short)  # Create the variable with type np.short
    for key, value in kwargs.items():
        setattr(var, key, value)  # Set attributes dynamically


def calculate_area_arcrad(xv, yv):
    """
    Calculate the area of grid cells in arc radians squared.

    Parameters:
        xv: NumPy array of longitude coordinates of the four vertices (in degrees), shape (4, n_cells).
        yv: NumPy array of latitude coordinates of the four vertices (in degrees), shape (4, n_cells).

    Returns:
        area_arcrad2: NumPy array of areas of the grid cells in arc radians squared, shape (n_cells,).
    """
    # Convert degrees to radians
    xv_rad = np.radians(xv)
    yv_rad = np.radians(yv)

    # Extract the four vertices in radians
    lon0, lon1, lon2, lon3 = xv_rad
    lat0, lat1, lat2, lat3 = yv_rad

    # Calculate the spherical excess using the vertices
    # Spherical excess formula: E = A + B + C - π
    # A, B, C are the angles of the spherical triangle
    # Split the quadrilateral into two triangles (0-1-2 and 0-2-3)

    # Triangle 1 (vertices 0, 1, 2)
    E1 = (
        np.sin(lat0) * np.sin(lat1) * np.cos(lon1 - lon0)
        + np.sin(lat1) * np.sin(lat2) * np.cos(lon2 - lon1)
        + np.sin(lat2) * np.sin(lat0) * np.cos(lon0 - lon2)
    )
    E1 = np.arccos(E1)

    # Triangle 2 (vertices 0, 2, 3)
    E2 = (
        np.sin(lat0) * np.sin(lat2) * np.cos(lon2 - lon0)
        + np.sin(lat2) * np.sin(lat3) * np.cos(lon3 - lon2)
        + np.sin(lat3) * np.sin(lat0) * np.cos(lon0 - lon3)
    )
    E2 = np.arccos(E2)

    # Total spherical excess
    spherical_excess = E1 + E2 - np.pi

    # Area in arc radians squared
    area_arcrad2 = abs(spherical_excess)

    return area_arcrad2


def create_variable(nc_file, name, dtype, dims, data, **kwargs):
    """
    Helper function to create a NetCDF variable.
    """
    var = nc_file.createVariable(name, dtype, dims, zlib=True, complevel=5)
    for key, value in kwargs.items():
        setattr(var, key, value)
    var[...] = data


def domain_1dNA(output_path, grid_data):
    """
    Save 1D domain data for the Daymet NA region.
    """
    formatted_date = datetime.now().strftime('%y%m%d')

    # Create the Txy2lonlat transformer
    geoxy_proj_str = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs"
    geoxyProj = CRS.from_proj4(geoxy_proj_str)
    lonlatProj = CRS.from_epsg(4326)
    Txy2lonlat = Transformer.from_proj(geoxyProj, lonlatProj, always_xy=True)

    # Extract the first time slice of the data (assuming time is the first dimension)
    data = grid_data["data"][0, :, :]  # Select the first time slice
    total_rows, total_cols = data.shape  # Get the shape of the 2D data array

    #create land gridcell mask, area, and landfrac (otherwise 0)
    masked = np.where(~np.isnan(data))
    landmask = np.where(~np.isnan(data), 1, 0)
    landfrac = landmask.astype(float)*1.0

    grid_ids = np.arange(total_rows * total_cols).reshape(data.shape)
    grid_id_arr = grid_ids[masked]
    landmask_arr = landmask[masked]
    landfrac_arr = landfrac[masked]
    lat_arr, lon_arr = grid_data["lat"][masked], grid_data["lon"][masked]

    XC, YC = np.meshgrid(grid_data["x_dim"], grid_data["y_dim"])  ## XC, YC are the locations of the center of gridcells in LCC
    # Calculate x_offset and y_offset from XC and YC
    x_offset = abs(XC[0, 1] - XC[0, 0])/2  # Half difference between adjacent x-coordinates
    y_offset = abs(YC[1, 0] - YC[0, 0])/2  # Half difference between adjacent y-coordinates

    XC_arr, YC_arr = XC[masked], YC[masked]
    area_arr = np.full(grid_id_arr.shape, 1.0)  # Based on data creation, area in km²
    #area_arcrad_arr = np.full(grid_id_arr.shape, 0.0)  # Placeholder for area in arcrad²

    # Calculate the four vertices of grid cells and store them in xv and yv
    xv, yv = calculate_vertices(Txy2lonlat, XC_arr, YC_arr, x_offset, y_offset)

    # Calculate the area of each grid cell in arc radians squared
    area_arcrad2_arr = calculate_area_arcrad(xv, yv)

    file_name = os.path.join(output_path, f'domain.lnd.Daymet_NA.1km.1d.c{formatted_date}.nc')
    print(f"Saving 1D domain file: {file_name}")

    with nc.Dataset(file_name, 'w', format='NETCDF4') as dst:
        dst.title = '1D domain file for the Daymet NA region'

        # Add the Lambert Conformal Conic projection variable
        create_projection_variable(
            dst,
            'lambert_conformal_conic',
            grid_mapping_name="lambert_conformal_conic",
            longitude_of_central_meridian=-100.0,
            latitude_of_projection_origin=42.5,
            false_easting=0.0,
            false_northing=0.0,
            standard_parallel=[25.0, 60.0],
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563
        )

        # Define dimensions
        dst.createDimension('ni', grid_id_arr.size)
        dst.createDimension('nj', 1)
        dst.createDimension('nv', 4)  # Add a new dimension for the 4 vertices

        create_variable(dst, 'gridID', np.int32, ('nj', 'ni'), grid_id_arr, long_name='Grid ID in the NA domain', 
            description='start from #0 at the upper left corner of the domain, covering all land and ocean gridcells')
        create_variable(dst, 'xc_lcc', np.float32, ('nj', 'ni'), XC_arr, long_name='x_coordinate (LCC) of grid cell center, inceasing from west to east', units='m')
        create_variable(dst, 'yc_lcc', np.float32, ('nj', 'ni'), YC_arr, long_name='y_coordinate (LCC) of grid cell center, decreasing from north to south', units='m')
        create_variable(dst, 'xc', np.float32, ('nj', 'ni'), lon_arr, long_name='Longitude of grid cell center, increasing from west to east', units='degrees_east')
        create_variable(dst, 'yc', np.float32, ('nj', 'ni'), lat_arr, long_name='Latitude of grid cell center, decreasing from north to south', units='degrees_north')
        create_variable(dst, 'xv', np.float32, ('nv','nj', 'ni'), xv, long_name='Longitude of grid cell vertices', units='degrees_east')
        create_variable(dst, 'yv', np.float32, ('nv', 'nj', 'ni'), yv, long_name='Latitude of grid cell vertices', units='degrees_north')

        create_variable(dst, 'mask', np.int32, ('nj', 'ni'), landmask_arr, long_name='Land mask', units='unitless')
        create_variable(dst, 'frac', np.float32, ('nj', 'ni'), landfrac_arr, long_name='Land fraction', units='unitless')
        create_variable(dst, 'area', np.float32, ('nj', 'ni'), area_arr, long_name='Area of grid cells (LCC)', units='km^2')
        create_variable(dst, 'area_arcrad', np.float32, ('nj', 'ni'), area_arcrad2_arr, long_name='Area of grid cells (radian)', units='radian^2')

def domain_2dNA(output_path, grid_data):
    """
    Save 2D domain data for the Daymet NA region.
    """
    formatted_date = datetime.now().strftime('%y%m%d')

    # Create the Txy2lonlat transformer
    geoxy_proj_str = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs"
    geoxyProj = CRS.from_proj4(geoxy_proj_str)
    lonlatProj = CRS.from_epsg(4326)
    Txy2lonlat = Transformer.from_proj(geoxyProj, lonlatProj, always_xy=True)

    # Extract the first time slice of the data (assuming time is the first dimension)
    data = grid_data["data"][0, :, :]  # Select the first time slice
    total_rows, total_cols = data.shape  # Get the shape of the 2D data array

    XC, YC = np.meshgrid(grid_data["x_dim"], grid_data["y_dim"])  ## XC, YC are the locations of the center of gridcells in LCC
    # Calculate x_offset and y_offset from XC and YC
    x_offset = abs(XC[0, 1] - XC[0, 0])/2  # Half difference between adjacent x-coordinates
    y_offset = abs(YC[1, 0] - YC[0, 0])/2  # Half difference between adjacent y-coordinates

    # Calculate the four vertices of grid cells and store them in xv and yv
    xv, yv = calculate_vertices(Txy2lonlat, XC, YC, x_offset, y_offset)

    # Calculate the area of each grid cell in arc radians squared
    area = np.full(data.shape, 1.0)  # Based on data creation, area in km²
    area_arcrad2 = calculate_area_arcrad(xv, yv)

    landmask = ~np.isnan(grid_data["data"][0]).astype(int)
    landfrac = landmask.astype(float)

    file_name = os.path.join(output_path, f'domain.lnd.Daymet_NA.1km.2d.c{formatted_date}.nc')
    print(f"Saving 2D domain file: {file_name}")

    with nc.Dataset(file_name, 'w', format='NETCDF4') as dst:
        dst.title = '2D domain file for the Daymet NA region'

        # Add the Lambert Conformal Conic projection variable
        create_projection_variable(
            dst,
            'lambert_conformal_conic',
            grid_mapping_name="lambert_conformal_conic",
            longitude_of_central_meridian=-100.0,
            latitude_of_projection_origin=42.5,
            false_easting=0.0,
            false_northing=0.0,
            standard_parallel=[25.0, 60.0],
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563
        )

        # Define dimensions
        dst.createDimension('ni', total_cols)
        dst.createDimension('nj', total_rows)
        dst.createDimension('nv', 4)  # Add a new dimension for the 4 vertices

        create_variable(dst, 'x_lcc', np.float32, ('ni'), grid_data["x_dim"], long_name='x_coordinate of grid cell center,increasing from west to east', units='m')
        create_variable(dst, 'y_lcc', np.float32, ('nj'), grid_data["y_dim"], long_name='y_coordinate of grid cell center, decreasing from north to south', units='m')
        create_variable(dst, 'xc', np.float32, ('nj', 'ni'), grid_data["lon"], long_name='Longitude of grid cell center', units='degrees_east')
        create_variable(dst, 'yc', np.float32, ('nj', 'ni'), grid_data["lat"], long_name='Latitude of grid cell center', units='degrees_north')
        create_variable(dst, 'xv', np.float32, ('nv', 'nj', 'ni'), xv, long_name='Longitude of grid cell vertices', units='degrees_east')
        create_variable(dst, 'yv', np.float32, ('nv', 'nj', 'ni'), yv, long_name='Latitude of grid cell vertices', units='degrees_north')
        create_variable(dst, 'mask', np.int32, ('nj', 'ni'), landmask, long_name='Land mask (1 means land)', units='unitless')
        create_variable(dst, 'frac', np.float32, ('nj', 'ni'), landfrac, long_name='Land fraction', units='unitless')
        create_variable(dst, 'area', np.float32, ('nj', 'ni'), area, long_name='Area of grid cells (LCC)', units='km^2')
        create_variable(dst, 'area_arcrad', np.float32, ('nj', 'ni'), area_arcrad2, long_name='Area of grid cells (radian)', units='radian^2')


def main():
    input_path = './'
    file_name = 'example_data.nc'
    output_path = input_path

    with nc.Dataset(file_name, 'r', format='NETCDF4') as src:
        #total_cols = src.dimensions['x'].size
        #total_rows = src.dimensions['y'].size
        x_dim, y_dim = src['x'][:], src['y'][:]  # center of gridcells (x, y)
        # move this to other place as they are temparary array
        #XC, YC = np.meshgrid(x_dim, y_dim)  ## XC, YC are the locations of the center of gridcells in LCC

        data = src['TBOT'][0:1, :, :]  # use for mask generation

        lon, lat = src['lon'][:, :], src['lat'][:, :] # center of gridcells (lon, lat)

        domain_data = {
            "lon": lon,
            "lat": lat,
            "x_dim": x_dim,
            "y_dim": y_dim,
            "data": data,
        }

    start = process_time()
    domain_2dNA(output_path, domain_data)
    print(f"2D domain data saved in {process_time() - start:.2f} seconds")

    start = process_time()
    domain_1dNA(output_path, domain_data)
    print(f"1D domain data saved in {process_time() - start:.2f} seconds")


if __name__ == '__main__':
    main()