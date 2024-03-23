
# utility to generate 1D and 2D domain
# use previous dataPartition module. Code need to be clean up

import os 
import math
import netCDF4 as nc
import numpy as np
from itertools import cycle
from time import process_time
from pyproj import Proj
from pyproj import Transformer
from pyproj import CRS
import datetime

def calculate_area(xv0, yv0, xv1, yv1, xv2, yv2, xv3, yv3):

    #Proj4: +proj=lcc +lon_0=-100 +lat_1=25 +lat_2=60 +k=1 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m  +no_defs
    geoxy_proj_str = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs"
    geoxyProj = CRS.from_proj4(geoxy_proj_str)
    # EPSG: 4326
    # Proj4: +proj=longlat +datum=WGS84 +no_defs
    lonlatProj = CRS.from_epsg(4326) # in lon/lat coordinates
    Txy2lonlat = Transformer.from_proj(geoxyProj, lonlatProj, always_xy=True)
    Tlonlat2xy = Transformer.from_proj(lonlatProj, geoxyProj, always_xy=True)
    
    # Step 1: Calculate the average value (Lat1) of the latitudes of two upper corners
    Lat1 = (yv0 + yv1) / 2

    # Step 2: Calculate the average value (Lat2) of the latitudes of two lower corners
    Lat2 = (yv2 + yv3) / 2

    # Step 3: Calculate the dLat = Lat1-Lat2
    dLat = Lat1 - Lat2

    # Step 4: Calculate the average value (Lon1) of the longitude of two left corners
    Lon1 = (xv0 + xv3) / 2

    # Step 5: Calculate the average value (Lon2) of the longitude of two right corners
    Lon2 = (xv1 + xv2) / 2

    # Step 6: Calculate the dLon = Lon1-Lon2
    dLon = Lon1 - Lon2

    # Step 7: Calculate the area
    area_arcad2 = abs((dLon / 180) * (dLat / 180))

    # Step 8: Calculate the area in LCC projection (km2)
    
    dX1, dY1 = Tlonlat2xy.transform(Lon1,dLat)
    dX2, dY2 = Tlonlat2xy.transform(Lon2,dLat)
    dX3, dY3 = Tlonlat2xy.transform(dLon,Lat1)
    dX4, dY4 = Tlonlat2xy.transform(dLon,Lat2)
    
    area_km2 = abs(((dX1 - dX2)/1000) * ((dY1 - dY2) / 1000))

    return area_arcad2, area_km2

def domain_info_1D(total_rows, total_cols, timesteps, data, lon, lat, XC, YC):
    total_gridcells = total_rows * total_cols
    grid_ids = np.linspace(0, total_gridcells-1, total_gridcells, dtype=int)
    grid_ids = grid_ids.reshape(lat.shape)

    """
    # create a mask for land grid_ids (1)
    mask = data[0]    # FSDS is in (time, Y, X) format
    mask = np.where(~np.isnan(mask), 1, 0)
    # Create a boolean mask where True indicates non-NaN values
    bool_mask = ~np.isnan(data[0])

    # create an flattened list of land gridID and reduce the size of gridIDs array
    grid_ids = grid_ids.reshape(total_cols,total_rows)
    grid_ids = np.multiply(mask,grid_ids)
    grid_ids = grid_ids[grid_ids != 0]

    # use the size of land gridcells to resize the FSDS matrix

    landcells = len(grid_ids)    
    
    lon = np.multiply(mask,lon)
    lon = lon[lon != 0]
    
    lat = np.multiply(mask,lat)
    lat = lat[lat != 0]

    XC = np.multiply(mask,XC)
    XC = XC[XC != 0]
    """
    # Create a boolean mask where True indicates non-NaN values
    bool_mask = ~np.isnan(data[0])
    #land_mask = np.where(~np.isnan(data[0]), 1, 0)
    # Apply the mask to the data data array
    grid_ids = grid_ids[bool_mask]
    lon = lon[bool_mask]
    lat = lat[bool_mask]
    XC = XC[bool_mask]
    YC = YC[bool_mask]

    print("domain info" + str(lon.shape) + str(lat.shape) + str(XC.shape) \
            + str(YC.shape) + str(bool_mask.shape) + str(grid_ids.shape))
    return grid_ids, data, lon, lat, XC, YC, bool_mask


def domain_save_1dTES(output_path, total_rows, total_cols, data, lon, lat, XC, YC):

    # This function uses (lon, lat) to create a domain for TES project
    # total_rows and total_cols are rows and cols from 2D Forcing dataset
    # lon, lat are 1D arrays
    # XC, YC are grid meshes that created using lon and Lat
   
    # Get current date
    current_date = datetime.datetime.now()

    # Format date as yymmdd
    date_string = current_date.strftime('%y%m%d')

    #Proj4: +proj=lcc +lon_0=-100 +lat_1=25 +lat_2=60 +k=1 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m  +no_defs
    geoxy_proj_str = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs"
    geoxyProj = CRS.from_proj4(geoxy_proj_str)
    # EPSG: 4326
    # Proj4: +proj=longlat +datum=WGS84 +no_defs
    lonlatProj = CRS.from_epsg(4326) # in lon/lat coordinates
    Txy2lonlat = Transformer.from_proj(geoxyProj, lonlatProj, always_xy=True)
    Tlonlat2xy = Transformer.from_proj(lonlatProj, geoxyProj, always_xy=True)

    XC_LCC,YC_LCC = Tlonlat2xy.transform(XC,YC)

    # add the gridcell IDs. and its x/y indices (will be used to create 2D output)
    total_gridcells = total_rows * total_cols
    grid_ids = np.linspace(0, total_gridcells-1, total_gridcells, dtype=int)
    grid_ids = grid_ids.reshape(total_rows,total_cols)
    grid_xids = np.indices(grid_ids.shape)[1]
    grid_yids = np.indices(grid_ids.shape)[0]

    print("shapes of gridid arrays", grid_ids.shape, grid_xids.shape, grid_yids.shape)
    
    #create land gridcell mask, area, and landfrac (otherwise 0)
    masked = np.where(~np.isnan(data[0]))
    landmask = np.where(~np.isnan(data[0]), 1, 0)
    landfrac = landmask.astype(float)*1.0
    
    # area in km2 --> in arcrad2
    area_km2 = 16.0 # 4km by 4km resolution
    '''offset = 2000.0 # distance (m) to the center of the grid
    side_km = math.sqrt(float(area_km2))
    lat[lat==90.0]=lat[lat==90.0]-0.00001
    lat[lat==-90.0]=lat[lat==-90.0]-0.00001
    kmratio_lon2lat = np.cos(np.radians(lat))
    re_km = 6371.22
    yscalar = side_km/(math.pi*re_km/180.0)
    xscalar = side_km/(math.pi*re_km/180.0*kmratio_lon2lat)
    area = xscalar*yscalar'''

    #  need to use the lon-lat bounding box to calculate the area
    #  Assume the resolution is fixed for all the cells
    lon_offset = abs((lon[0]-lon[1])/2)
    lat_offset = abs((lat[0]-lat[1])/2)
    
    # 2d --> 1d, with masked only
    grid_id_arr = grid_ids[masked]
    mask_arr = landmask[masked]
    landfrac_arr = landfrac[masked]
    grid_xids_arr = grid_xids[masked]
    grid_yids_arr = grid_yids[masked]
    XC_arr = XC[masked]
    YC_arr = YC[masked]
    XC_LCC_arr = XC_LCC[masked]
    YC_LCC_arr = YC_LCC[masked]
    
    file_name = output_path + 'domain.lnd.Daymet4.TESSFA.4km.1d.c' + date_string + '.nc'
    print("The domain file is " + file_name)

    # Open a new NetCDF file to write the domain information. For format, you can choose from
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    w_nc_fid = nc.Dataset(file_name, 'w', format='NETCDF4')
    w_nc_fid.title = '1D domain file for the Daymet NA region'

    # Create new dimensions of new coordinate system
    x_dim = w_nc_fid.createDimension('x', total_cols)
    y_dim = w_nc_fid.createDimension('y', total_rows)
    dst_var = w_nc_fid.createVariable('x', np.float32, ('x'))
    dst_var.units = "degree"
    dst_var.long_name = "x coordinate of projection"
    dst_var.standard_name = "projection_x_coordinate"
    w_nc_fid['x'][...] = np.copy(XC[0,:])
    dst_var = w_nc_fid.createVariable('y', np.float32, ('y'))
    dst_var.units = "degree"
    dst_var.long_name = "y coordinate of projection"
    dst_var.standard_name = "projection_y_coordinate"
    w_nc_fid['y'][...] = np.copy(YC[:,0])

    # create the gridIDs, lon, and lat variable
    ni_dim = w_nc_fid.createDimension('ni', grid_id_arr.size)
    nj_dim = w_nc_fid.createDimension('nj', 1)
    nv_dim = w_nc_fid.createDimension('nv', 4)

    w_nc_var = w_nc_fid.createVariable('gridID', np.int32, ('nj','ni'))
    w_nc_var.long_name = 'gridId in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner of the domain, covering all land and ocean gridcells" 
    w_nc_fid.variables['gridID'][...] = grid_id_arr

    w_nc_var = w_nc_fid.createVariable('gridXID', np.int32, ('nj','ni'))
    w_nc_var.long_name = 'gridId x in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner and from west to east of the domain, with gridID=gridXID+gridYID*x_dim" 
    w_nc_fid.variables['gridXID'][...] = grid_xids_arr

    w_nc_var = w_nc_fid.createVariable('gridYID', np.int32, ('nj','ni'))
    w_nc_var.long_name = 'gridId y in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner and from north to south of the domain, with gridID=gridXID+gridYID*y_dim" 
    w_nc_fid.variables['gridYID'][...] = grid_yids_arr
    

    w_nc_var = w_nc_fid.createVariable('xc', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'longitude of land gridcell center (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc'][...] = XC_arr
        
    w_nc_var = w_nc_fid.createVariable('yc', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'latitude of land gridcell center (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"        
    w_nc_fid.variables['yc'][...] = YC_arr
        
    # create the XC, YC variable
    w_nc_var = w_nc_fid.createVariable('xc_LCC', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'X of land gridcell center (Lambert Conformal Conic), increasing from west to east'
    w_nc_var.units = "m"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc_LCC'][...] = XC_LCC_arr
        
    w_nc_var = w_nc_fid.createVariable('yc_LCC', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'Y of land gridcell center (Lambert Conformal Conic), decreasing from north to south'
    w_nc_var.units = "m"        
    w_nc_fid.variables['yc_LCC'][...] = YC_LCC_arr

    #
    w_nc_var = w_nc_fid.createVariable('xv', np.float32, ('nv','nj','ni'))
    w_nc_var.long_name = 'longitude of land gridcell verticles (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    w_nc_var = w_nc_fid.createVariable('yv', np.float32, ('nv','nj','ni'))
    w_nc_var.long_name = 'latitude of land gridcell verticles (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"

    xv0,yv0 = (XC_arr-lon_offset,YC_arr + lat_offset)
    w_nc_fid.variables['xv'][0,...] = xv0
    w_nc_fid.variables['yv'][0,...] = yv0
    xv1,yv1 = (XC_arr+ lon_offset,YC_arr+ lat_offset)
    w_nc_fid.variables['xv'][1,...] = xv1
    w_nc_fid.variables['yv'][1,...] = yv1
    xv2,yv2 = (XC_arr+ lon_offset,YC_arr- lat_offset)
    w_nc_fid.variables['xv'][2,...] = xv2
    w_nc_fid.variables['yv'][2,...] = yv2
    xv3,yv3 = (XC_arr- lon_offset,YC_arr- lat_offset)
    w_nc_fid.variables['xv'][3,...] = xv3
    w_nc_fid.variables['yv'][3,...] = yv3

    area_arcad2, area_km2 = calculate_area(xv0, yv0, xv1, yv1, xv2, yv2, xv3, yv3)

    w_nc_var = w_nc_fid.createVariable('area', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'Area of land gridcells'
    w_nc_var.coordinate = 'xc yc' 
    w_nc_var.units = "arcad^2"
    w_nc_fid.variables['area'][...] = area_arcad2

    w_nc_var = w_nc_fid.createVariable('area_LCC', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'Area of land gridcells (Lambert Conformal Conic)'
    w_nc_var.coordinate = 'xc_LCC yc_LCC' 
    w_nc_var.units = "km^2"
    w_nc_fid.variables['area_LCC'][...] = area_km2

    w_nc_var = w_nc_fid.createVariable('mask', np.int32, ('nj','ni'))
    w_nc_var.long_name = 'mask of land gridcells (1 means land)'
    w_nc_var.units = "unitless"        
    w_nc_fid.variables['mask'][...] = mask_arr

    w_nc_var = w_nc_fid.createVariable('frac', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'fraction of land gridcell that is active'
    w_nc_var.coordinate = 'xc yc' 
    w_nc_var.units = "unitless" ;        
    w_nc_fid.variables['frac'][...] = landfrac_arr

    w_nc_var = w_nc_fid.createVariable('lambert_conformal_conic', np.short)
    w_nc_var.grid_mapping_name = "lambert_conformal_conic"
    w_nc_var.longitude_of_central_meridian = -100. 
    w_nc_var.latitude_of_projection_origin = 42.5         
    w_nc_var.false_easting = 0. 
    w_nc_var.false_northing = 0.
    w_nc_var.standard_parallel = 25., 60.          
    w_nc_var.semi_major_axis = 6378137.
    w_nc_var.inverse_flattening = 298.257223563

    w_nc_fid.close()  # close the new file  


def domain_save_2dTES(output_path, total_rows, total_cols, data, lon, lat, XC, YC):

    # It appears that lon/lat in original GSWP3/daymet4 dataset are questionable (unclear of what datanum used). 
    # Instead here redo lon/lat from XC/YC
    #Proj4: +proj=lcc +lon_0=-100 +lat_1=25 +lat_2=60 +k=1 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m  +no_defs
    geoxy_proj_str = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs"
    geoxyProj = CRS.from_proj4(geoxy_proj_str)
    # EPSG: 4326
    # Proj4: +proj=longlat +datum=WGS84 +no_defs
    lonlatProj = CRS.from_epsg(4326) # in lon/lat coordinates
    Txy2lonlat = Transformer.from_proj(geoxyProj, lonlatProj, always_xy=True)
    Tlonlat2xy = Transformer.from_proj(lonlatProj, geoxyProj, always_xy=True)

    # Get current date
    current_date = datetime.datetime.now()

    # Format date as yymmdd
    date_string = current_date.strftime('%y%m%d')

    XC_LCC, YC_LCC = Tlonlat2xy.transform(XC,YC)
    
    # add the gridcell IDs.
    total_gridcells = total_rows * total_cols
    grid_ids = np.linspace(0, total_gridcells-1, total_gridcells, dtype=int)
    grid_ids = grid_ids.reshape(total_rows,total_cols)
    grid_xids = np.indices(grid_ids.shape)[1]
    grid_yids = np.indices(grid_ids.shape)[0]

    #create land gridcell mask, area, and landfrac (otherwise 0)
    mask = np.where(~np.isnan(data[0]), 1, 0)

    # area in km2 --> in arcrad2
    '''
    area_km2 = 16.0  # 4km by 4 km resolution
    side_km = math.sqrt(float(area_km2))
    #lat[lat==90.0]=lat[lat==90.0]-0.00001
    #lat[lat==-90.0]=lat[lat==-90.0]-0.00001
    kmratio_lon2lat = np.cos(np.radians(lat))
    re_km = 6378.137
    yscalar = side_km/(math.pi*re_km/180.0)
    xscalar = side_km/(math.pi*re_km/180.0*kmratio_lon2lat)
    area = xscalar*yscalar
    '''
    
    #  need to use the lon-lat bounding box to calculate the area
    lon_offset = abs((lon[0]-lon[1])/2)
    lat_offset = abs((lat[0]-lat[1])/2)

    landfrac = mask.astype(float)*1.0

    file_name = output_path + 'domain.lnd.Daymet4.TESSFA.4km.2d.c'+ date_string + '.nc'
    print("The 2D domain file is " + file_name)

    # Open a new NetCDF file to write the domain information. For format, you can choose from
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    w_nc_fid = nc.Dataset(file_name, 'w', format='NETCDF4')
    w_nc_fid.title = '2D domain file for the Daymet NA region'

    # create the gridIDs, lon, and lat variable
    x_dim = w_nc_fid.createDimension('ni', total_cols)
    y_dim = w_nc_fid.createDimension('nj', total_rows)
    v_dim = w_nc_fid.createDimension('nv', 4)

    # Create new dimensions of new coordinate system
    dst_var = w_nc_fid.createVariable('ni', np.float32, ('ni'))
    dst_var.units = "degree"
    dst_var.long_name = "x coordinate of projection"
    dst_var.standard_name = "projection_x_coordinate"
    w_nc_fid['ni'][...] = np.copy(XC[0,:])
    dst_var = w_nc_fid.createVariable('nj', np.float32, ('nj'))
    dst_var.units = "m"
    dst_var.long_name = "y coordinate of projection"
    dst_var.standard_name = "projection_y_coordinate"
    w_nc_fid['nj'][...] = np.copy(YC[:,0])

    w_nc_var = w_nc_fid.createVariable('gridID', np.int32, ('nj','ni'))
    w_nc_var.long_name = 'gridId in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner of the domain and row by row, covering all land and ocean gridcells. " \
                          "So gridID=xid+yid*ni" 
    w_nc_fid.variables['gridID'][:] = grid_ids 

    w_nc_var = w_nc_fid.createVariable('xc', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'longitude of land gridcell center (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc'][:] = XC
        
    w_nc_var = w_nc_fid.createVariable('yc', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'latitude of land gridcell center (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"        
    w_nc_fid.variables['yc'][:] = YC

    w_nc_var = w_nc_fid.createVariable('xv', np.float32, ('nv','nj','ni'))
    w_nc_var.long_name = 'longitude of land gridcell verticles (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    w_nc_var = w_nc_fid.createVariable('yv', np.float32, ('nv','nj','ni'))
    w_nc_var.long_name = 'latitude of land gridcell verticles (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"

    # calculate the bounding box and then the area in arcrad^2 and km2
    xv0,yv0 = (XC-lon_offset,YC + lat_offset)  # upper-left corner
    w_nc_fid.variables['xv'][0,...] = xv0
    w_nc_fid.variables['yv'][0,...] = yv0
    xv1,yv1 = (XC+ lon_offset,YC+ lat_offset)  # upper-right corner
    w_nc_fid.variables['xv'][1,...] = xv1
    w_nc_fid.variables['yv'][1,...] = yv1
    xv2,yv2 = (XC+ lon_offset,YC- lat_offset)  # lower-right corner
    w_nc_fid.variables['xv'][2,...] = xv2
    w_nc_fid.variables['yv'][2,...] = yv2
    xv3,yv3 = (XC- lon_offset,YC- lat_offset)  # lower-left corner
    w_nc_fid.variables['xv'][3,...] = xv3
    w_nc_fid.variables['yv'][3,...] = yv3

    area_arcad2, area_km2 = calculate_area(xv0, yv0, xv1, yv1, xv2, yv2, xv3, yv3)
    
    # create the XC_LCC, YC_LCC variable
    w_nc_var = w_nc_fid.createVariable('xc_LCC', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'X of land gridcell center (Lambert Conformal Conic), increasing from west to east'
    w_nc_var.units = "m"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc_LCC'][:] = XC_LCC
        
    w_nc_var = w_nc_fid.createVariable('yc_LCC', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'Y of land gridcell center (Lambert Conformal Conic), decreasing from north to south'
    w_nc_var.units = "m"        
    w_nc_fid.variables['yc_LCC'][:] = YC_LCC

    w_nc_var = w_nc_fid.createVariable('area', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'Area of land gridcells'
    w_nc_var.coordinate = 'xc yc' 
    w_nc_var.units = "arcad^2"
    w_nc_fid.variables['area'][...] = area_arcad2

    w_nc_var = w_nc_fid.createVariable('area_LCC', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'Area of land gridcells (Lambert Conformal Conic)'
    w_nc_var.coordinate = 'xc_LCC yc_LCC' 
    w_nc_var.units = "km^2"
    w_nc_fid.variables['area_LCC'][:] = area_km2

    w_nc_var = w_nc_fid.createVariable('mask', np.int32, ('nj','ni'))
    w_nc_var.long_name = 'mask of land gridcells (1 means land)'
    w_nc_var.units = "unitless"        
    w_nc_fid.variables['mask'][:] = mask  

    w_nc_var = w_nc_fid.createVariable('frac', np.float32, ('nj','ni'))
    w_nc_var.long_name = 'fraction of land gridcell that is active'
    w_nc_var.coordinate = 'xc yc' 
    w_nc_var.units = "unitless" ;        
    w_nc_fid.variables['frac'][:] = landfrac  


    w_nc_var = w_nc_fid.createVariable('lambert_conformal_conic', np.short)
    w_nc_var.grid_mapping_name = "lambert_conformal_conic"
    w_nc_var.longitude_of_central_meridian = -100. 
    w_nc_var.latitude_of_projection_origin = 42.5         
    w_nc_var.false_easting = 0. 
    w_nc_var.false_northing = 0.
    w_nc_var.standard_parallel = 25., 60.          
    w_nc_var.semi_major_axis = 6378137.
    w_nc_var.inverse_flattening = 298.257223563

    w_nc_fid.close()  # close the new file  

    
input_path= './'
file_name = 'TES_TBOT0.nc'
output_path = input_path
number_of_subdomains = 1
timesteps = 1
var_name = 'TBOT'
period= "2014"


    #print('number of subdomains: ('+ str(number_of_subdomains) + \
    #      '), timeseries: ('+ str(i_timesteps) + '), file_name: (' + file_name + \
    #      '), input_path : (' + input_path + ')')

start = process_time()
file_name = input_path + file_name

r_nc_fid = nc.Dataset(file_name, 'r', format='NETCDF4')

total_cols = r_nc_fid.dimensions['lon'].size
total_rows = r_nc_fid.dimensions['lat'].size
total_timesteps = r_nc_fid.dimensions['time'].size
lon = r_nc_fid['lon']
lat = r_nc_fid['lat']
XC, YC = np.meshgrid(lon, lat)  # the array is (y,x) to match the mask
    #YX = np.array([Y, X])
    #print('total timesteps is :' + str(total_timesteps))

# Round to 2 decimal places
XC = np.around(XC, 8)
YC = np.around(YC, 8)

if timesteps == -1:
    timesteps = total_timesteps
data = r_nc_fid[var_name][0:timesteps, :, :] # read (timestep, y, x) format

print(lon.shape, lat.shape, XC.shape, YC.shape) 

end = process_time()
print("Reading " + file_name + " takes  {}".format(end-start))

start = process_time()
domain_save_2dTES(output_path, total_rows, total_cols, data, lon, lat, XC, YC)
end = process_time()
print("Saving 2D domain data takes {}".format(end-start))
    

start = process_time() 
domain_save_1dTES(output_path, total_rows, total_cols, data, lon, lat, XC, YC)
end = process_time()

print("Saving 1D domain data takes {}".format(end-start))

