
# utility (V2) to generate 1D and 2D domain for Daymet Domain

import os 
import math
import netCDF4 as nc
import numpy as np
from itertools import cycle
from time import process_time
from pyproj import Proj
from pyproj import Transformer
from pyproj import CRS
from datetime import datetime



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
      

def domain_save_1dNA(output_path, total_rows, total_cols, data, lon, lat, XC, YC):

    # Get current date
    current_date = datetime.now()
    # Format date to mmddyyyy
    formatted_date = current_date.strftime('%y%m%d')

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

    lon,lat = Txy2lonlat.transform(XC,YC)

    # add the gridcell IDs. and its x/y indices
    total_gridcells = total_rows * total_cols
    grid_ids = np.linspace(0, total_gridcells-1, total_gridcells, dtype=int)
    grid_ids = grid_ids.reshape(lat.shape)
    grid_xids = np.indices(grid_ids.shape)[1]
    grid_yids = np.indices(grid_ids.shape)[0]


    #create land gridcell mask, area, and landfrac (otherwise 0)
    masked = np.where(~np.isnan(data[0]))
    landmask = np.where(~np.isnan(data[0]), 1, 0)
    landfrac = landmask.astype(float)*1.0
    
    # area in km2 --> in arcrad2
    area_km2 = 1.0 # This is by default
    side_km = math.sqrt(float(area_km2))
    offset = side_km/2 * 1000  # convert to meter
    
    lat[lat==90.0]=lat[lat==90.0]-0.00001
    lat[lat==-90.0]=lat[lat==-90.0]-0.00001
    kmratio_lon2lat = np.cos(np.radians(lat))
    re_km = 6371.22
    yscalar = side_km/(math.pi*re_km/180.0)
    xscalar = side_km/(math.pi*re_km/180.0*kmratio_lon2lat)
    area = xscalar*yscalar
    
    #x_offset = abs((lon[0]-lon[1])/2)
    #y_offset = abs((lat[0]-lat[1])/2)
    
    # 2d --> 1d, with masked only
    grid_id_arr = grid_ids[masked]
    grid_xids_arr = grid_xids[masked]
    grid_yids_arr = grid_yids[masked]
    mask_arr = landmask[masked]
    landfrac_arr = landfrac[masked]
    lat_arr = lat[masked]
    lon_arr = lon[masked]
    XC_arr = XC[masked]
    YC_arr = YC[masked]

    file_name = output_path + 'domain.lnd.Daymet_NA.1km.1d.c'+ formatted_date + '.nc'
    print("The domain file is " + file_name)

    # Open a new NetCDF file to write the domain information. For format, you can choose from
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    w_nc_fid = nc.Dataset(file_name, 'w', format='NETCDF4')
    w_nc_fid.title = '1D domain file for the Daymet NA region'

    # Create new dimensions of new coordinate system
    x_dim = w_nc_fid.createDimension('x', total_cols)
    y_dim = w_nc_fid.createDimension('y', total_rows)
    dst_var = w_nc_fid.createVariable('x', np.float64, ('x'), zlib=True, complevel=5)
    dst_var.units = "m"
    dst_var.long_name = "x coordinate of projection"
    dst_var.standard_name = "projection_x_coordinate"
    w_nc_fid['x'][...] = np.copy(XC[0,:])
    dst_var = w_nc_fid.createVariable('y', np.float64, ('y'), zlib=True, complevel=5)
    dst_var.units = "m"
    dst_var.long_name = "y coordinate of projection"
    dst_var.standard_name = "projection_y_coordinate"
    w_nc_fid['y'][...] = np.copy(YC[:,0])
    w_nc_var = w_nc_fid.createVariable('lon', np.float64, ('y','x'), zlib=True, complevel=5)
    w_nc_var.long_name = 'longitude of 2D land gridcell center (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    w_nc_fid.variables['lon'][...] = lon
    w_nc_var = w_nc_fid.createVariable('lat', np.float64, ('y','x'), zlib=True, complevel=5)
    w_nc_var.long_name = 'latitude of 2D land gridcell center (GCS_WGS_84), increasing from south to north'
    w_nc_var.units = "degrees_north"
    w_nc_fid.variables['lat'][...] = lat

    # create the gridIDs, lon, and lat variable
    ni_dim = w_nc_fid.createDimension('ni', grid_id_arr.size)
    nj_dim = w_nc_fid.createDimension('nj', 1)
    nv_dim = w_nc_fid.createDimension('nv', 4)

    w_nc_var = w_nc_fid.createVariable('gridID', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'gridId in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner of the domain, covering all land and ocean gridcells" 
    w_nc_fid.variables['gridID'][...] = grid_id_arr

    w_nc_var = w_nc_fid.createVariable('gridXID', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'gridId x in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner and from west to east of the domain, with gridID=gridXID+gridYID*x_dim" 
    w_nc_fid.variables['gridXID'][...] = grid_xids_arr

    w_nc_var = w_nc_fid.createVariable('gridYID', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'gridId y in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner and from north to south of the domain, with gridID=gridXID+gridYID*y_dim" 
    w_nc_fid.variables['gridYID'][...] = grid_yids_arr

    w_nc_var = w_nc_fid.createVariable('xc', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'longitude of land gridcell center (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc'][...] = lon_arr
        
    w_nc_var = w_nc_fid.createVariable('yc', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'latitude of land gridcell center (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"        
    w_nc_fid.variables['yc'][...] = lat_arr
        
    # create the XC, YC variable
    w_nc_var = w_nc_fid.createVariable('xc_LCC', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'X of land gridcell center (Lambert Conformal Conic), increasing from west to east'
    w_nc_var.units = "m"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc_LCC'][...] = XC_arr
        
    w_nc_var = w_nc_fid.createVariable('yc_LCC', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'Y of land gridcell center (Lambert Conformal Conic), decreasing from north to south'
    w_nc_var.units = "m"        
    w_nc_fid.variables['yc_LCC'][...] = YC_arr

    #
    w_nc_var = w_nc_fid.createVariable('xv', np.float64, ('nv','nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'longitude of land gridcell verticles (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    w_nc_var = w_nc_fid.createVariable('yv', np.float64, ('nv','nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'latitude of land gridcell verticles (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"

    xv0,yv0 = Txy2lonlat.transform(XC_arr-offset,YC_arr+offset)
    w_nc_fid.variables['xv'][0,...] = xv0
    w_nc_fid.variables['yv'][0,...] = yv0
    xv1,yv1 = Txy2lonlat.transform(XC_arr+offset,YC_arr+offset)
    w_nc_fid.variables['xv'][1,...] = xv1
    w_nc_fid.variables['yv'][1,...] = yv1
    xv2,yv2 = Txy2lonlat.transform(XC_arr+offset,YC_arr-offset)
    w_nc_fid.variables['xv'][2,...] = xv2
    w_nc_fid.variables['yv'][2,...] = yv2
    xv3,yv3 = Txy2lonlat.transform(XC_arr-offset,YC_arr-offset)
    w_nc_fid.variables['xv'][3,...] = xv3
    w_nc_fid.variables['yv'][3,...] = yv3


    #area_arcad2, area_km2 = calculate_area(xv0, yv0, xv1, yv1, xv2, yv2, xv3, yv3)
    
    w_nc_var = w_nc_fid.createVariable('area', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'Area of land gridcells'
    w_nc_var.coordinate = 'xc yc' 
    w_nc_var.units = "radian^2"
    w_nc_fid.variables['area'][...] = area

    w_nc_var = w_nc_fid.createVariable('area_LCC', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'Area of land gridcells (Lambert Conformal Conic)'
    w_nc_var.coordinate = 'xc yc' 
    w_nc_var.units = "km^2"
    w_nc_fid.variables['area_LCC'][...] = area_km2

    w_nc_var = w_nc_fid.createVariable('mask', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'mask of land gridcells (1 means land)'
    w_nc_var.units = "unitless"        
    w_nc_fid.variables['mask'][...] = mask_arr

    w_nc_var = w_nc_fid.createVariable('frac', np.float32, ('nj','ni'), zlib=True, complevel=5)
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


def domain_save_2dNA(output_path, total_rows, total_cols, data, lon, lat, XC, YC):

    # Get current date
    current_date = datetime.now()
    # Format date to mmddyyyy
    formatted_date = current_date.strftime('%y%m%d')

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

    lon,lat = Txy2lonlat.transform(XC,YC)
    #
    
    # add the gridcell IDs.
    total_gridcells = total_rows * total_cols
    grid_ids = np.linspace(0, total_gridcells-1, total_gridcells, dtype=int)
    grid_ids = grid_ids.reshape(lat.shape)
    grid_xids = np.indices(grid_ids.shape)[1]
    grid_yids = np.indices(grid_ids.shape)[0]

    #create land gridcell mask, area, and landfrac (otherwise 0)
    mask = np.where(~np.isnan(data[0]), 1, 0)
    
    # area in km2 --> in arcrad2
    area_km2 = 1.0
    side_km = math.sqrt(float(area_km2))
    offset = side_km /2 
    
    '''lat[lat==90.0]=lat[lat==90.0]-0.00001
    lat[lat==-90.0]=lat[lat==-90.0]-0.00001
    kmratio_lon2lat = np.cos(np.radians(lat))
    re_km = 6378.137
    yscalar = side_km/(math.pi*re_km/180.0)
    xscalar = side_km/(math.pi*re_km/180.0*kmratio_lon2lat)
    area = xscalar*yscalar'''
    
    landfrac = mask.astype(float)*1.0

    file_name = output_path + 'domain.lnd.Daymet_NA.1km.2d.c'+ formatted_date + '.nc'
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
    dst_var = w_nc_fid.createVariable('ni', np.float64, ('ni'), zlib=True, complevel=5)
    dst_var.units = "m"
    dst_var.long_name = "x coordinate of projection"
    dst_var.standard_name = "projection_x_coordinate"
    w_nc_fid['ni'][...] = np.copy(XC[0,:])
    dst_var = w_nc_fid.createVariable('nj', np.float64, ('nj'), zlib=True, complevel=5)
    dst_var.units = "m"
    dst_var.long_name = "y coordinate of projection"
    dst_var.standard_name = "projection_y_coordinate"
    w_nc_fid['nj'][...] = np.copy(YC[:,0])

    w_nc_var = w_nc_fid.createVariable('gridID', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'gridId in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner of the domain and row by row, covering all land and ocean gridcells. " \
                          "So gridID=xid+yid*ni" 
    w_nc_fid.variables['gridID'][:] = grid_ids 
    w_nc_var = w_nc_fid.createVariable('gridXID', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'gridId x in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner and from west to east of the domain, with gridID=gridXID+gridYID*ni" 
    w_nc_fid.variables['gridXID'][...] = grid_xids

    w_nc_var = w_nc_fid.createVariable('gridYID', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'gridId y in the NA domain'
    w_nc_var.decription = "start from #0 at the upper left corner and from north to south of the domain, with gridID=gridXID+gridYID*ni" 
    w_nc_fid.variables['gridYID'][...] = grid_yids
    

    w_nc_var = w_nc_fid.createVariable('xc', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'longitude of land gridcell center (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc'][:] = lon
        
    w_nc_var = w_nc_fid.createVariable('yc', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'latitude of land gridcell center (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"        
    w_nc_fid.variables['yc'][:] = lat

    w_nc_var = w_nc_fid.createVariable('xv', np.float64, ('nv','nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'longitude of land gridcell verticles (GCS_WGS_84), increasing from west to east'
    w_nc_var.units = "degrees_east"
    w_nc_var = w_nc_fid.createVariable('yv', np.float64, ('nv','nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'latitude of land gridcell verticles (GCS_WGS_84), decreasing from north to south'
    w_nc_var.units = "degrees_north"

    xv0,yv0 = Txy2lonlat.transform(XC-offset,YC+offset)
    w_nc_fid.variables['xv'][0,...] = xv0
    w_nc_fid.variables['yv'][0,...] = yv0
    xv1,yv1 = Txy2lonlat.transform(XC+offset,YC+offset)
    w_nc_fid.variables['xv'][1,...] = xv1
    w_nc_fid.variables['yv'][1,...] = yv1
    xv2,yv2 = Txy2lonlat.transform(XC+offset,YC-offset)
    w_nc_fid.variables['xv'][2,...] = xv2
    w_nc_fid.variables['yv'][2,...] = yv2
    xv3,yv3 = Txy2lonlat.transform(XC-offset,YC-offset)
    w_nc_fid.variables['xv'][3,...] = xv3
    w_nc_fid.variables['yv'][3,...] = yv3

    area_arcad2, area_km2 = calculate_area(xv0, yv0, xv1, yv1, xv2, yv2, xv3, yv3)
    
    # create the XC, YC variable
    w_nc_var = w_nc_fid.createVariable('xc_LCC', np.float64, ('nj','ni'),zlib=True, complevel=5)
    w_nc_var.long_name = 'X of land gridcell center (Lambert Conformal Conic), increasing from west to east'
    w_nc_var.units = "m"
    #w_nc_var.bounds = "xv" ;    
    w_nc_fid.variables['xc_LCC'][:] = XC
        
    w_nc_var = w_nc_fid.createVariable('yc_LCC', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'Y of land gridcell center (Lambert Conformal Conic), decreasing from north to south'
    w_nc_var.units = "m"        
    w_nc_fid.variables['yc_LCC'][:] = YC

    w_nc_var = w_nc_fid.createVariable('area', np.float64, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'Area of land gridcells (Lambert Conformal Conic)'
    w_nc_var.coordinate = 'xc yc' 
    w_nc_var.units = "radian^2"
    w_nc_fid.variables['area'][:] = area_arcad2

    w_nc_var = w_nc_fid.createVariable('mask', np.int32, ('nj','ni'), zlib=True, complevel=5)
    w_nc_var.long_name = 'mask of land gridcells (1 means land)'
    w_nc_var.units = "unitless"        
    w_nc_fid.variables['mask'][:] = mask  

    w_nc_var = w_nc_fid.createVariable('frac', np.float64, ('nj','ni'), zlib=True, complevel=5)
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


    
def main():
    """
    args = sys.argv[1:]
    input_path = args[0]
    output_path = args[1]
    number_of_subdomains = int(arg[2])
    i_timesteps = int(arg[3])
    """
    
    input_path= './'
    file_name = 'clmforc.Daymet4.1km.TBOT.2014-01.nc'
    output_path = input_path
    number_of_subdomains = 1
    i_timesteps = 1
    var_name = 'TBOT'
    period= "2014"


    # Open a new NetCDF file to write the data to. For format, you can choose from
    # 'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', and 'NETCDF4'
    r_nc_fid = nc.Dataset(file_name, 'r', format='NETCDF4')

    total_cols = r_nc_fid.dimensions['x'].size
    total_rows = r_nc_fid.dimensions['y'].size
    total_timesteps = r_nc_fid.dimensions['time'].size
    x_dim = r_nc_fid['x']
    y_dim = r_nc_fid['y']
    XC, YC = np.meshgrid(x_dim, y_dim)  # the array is (y,x) to match the mask

    data = r_nc_fid[var_name][0:1, :, :] # read (timestep, y, x) format


    '''geoxy_proj_str = "+proj=lcc +lon_0=-100 +lat_0=42.5 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +R=6378137 +f=298.257223563 +units=m +no_defs"
    geoxyProj = CRS.from_proj4(geoxy_proj_str)
    # EPSG: 4326
    # Proj4: +proj=longlat +datum=WGS84 +no_defs
    lonlatProj = CRS.from_epsg(4326) # in lon/lat coordinates
    Txy2lonlat = Transformer.from_proj(geoxyProj, lonlatProj, always_xy=True)
    Tlonlat2xy = Transformer.from_proj(lonlatProj, geoxyProj, always_xy=True)

    lon,lat = Txy2lonlat.transform(XC,YC)'''

    
    lon = r_nc_fid['lon'][:,:]  # read lon(y, x) format
    lat = r_nc_fid['lat'][:,:]  # read lat(y, x) format

    print(lon.shape, lat.shape, XC.shape, YC.shape) 


    start = process_time()
    domain_save_2dNA(output_path, total_rows, total_cols, data, lon, lat, XC, YC)
    end = process_time()
    print("Saving 2D domain data takes {}".format(end-start))
    
    start = process_time() 
    domain_save_1dNA(output_path, total_rows, total_cols, data, lon, lat, XC, YC)
    end = process_time()
    print("Saving 1D domain data takes {}".format(end-start))


if __name__ == '__main__':
    main()

    
