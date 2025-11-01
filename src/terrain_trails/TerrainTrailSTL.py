# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:50:30 2021

@author: Jeremy Koether
"""


from typing import Any, Optional
import time as time
import os as os
import glob as glob
import numpy as np
import trimesh as tm
import gpxpy as gpxpy
from scipy import ndimage
import shapely as shp
from shapely.plotting import patch_from_polygon
import matplotlib.pyplot as plt
from importlib import resources

from .utils.dem_utils import Dem
from .utils.cord_utils import *
from .osm_queries import *
from .meshing import *
from .tiling import print_scaling_tiled


def print_scaling(dem: Dem, Boundary: np.ndarray, print_size: list[float]) -> tuple[float, np.ndarray, shp.geometry.Polygon]:
    """
    Determine the largest size that can be printed with a single print.
    """
    x = dem.lon
    y = dem.lat

    corner = np.stack((np.min(x), np.min(y)))

    p = cord2dist(Boundary, corner=corner)

    print_angle = np.linspace(0, np.pi / 2, 181)
    scale = np.zeros(print_angle.shape[0])

    for i in range(print_angle.shape[0]):
        # Rotated polygon:
        rp = shp.affinity.rotate(shp.geometry.Polygon(p), print_angle[i], use_radians=True)
        scale[i] = min(
            [
                print_size[0] / (rp.bounds[2] - rp.bounds[0]),
                print_size[1] / (rp.bounds[3] - rp.bounds[1]),
            ]
        )

    print("print angle: {0:.2f} deg".format(print_angle[np.argmax(scale)] * 180 / np.pi))

    scale = scale.max()  # Use angle that allows the largest print.
    x, y = cord2dist(x=x, y=y, corner=corner)
    print(
        "DEM resolution: {0:.2f}x{1:.2f} mm".format(
            ((x.max() - x.min()) / (x.shape[0] - 1) * scale),
            (y.max() - y.min()) / (y.shape[0] - 1) * scale,
        )
    )
    edge_poly = shp.geometry.Polygon(p * scale)
    dem.scale_factor = scale
    return scale, corner, edge_poly


def draw_polygon(ax: plt.Axes, ply: shp.geometry.base.BaseGeometry, **kwargs) -> None:
    """
    Draw a polygon or multipolygon on a matplotlib axis.
    """
    if isinstance(ply, MultiPolygon):
        for p in ply.geoms:
            draw_polygon(ax, p, **kwargs)
    elif isinstance(ply, Polygon) :
        patch = patch_from_polygon(ply, **kwargs)
        ax.add_patch(patch)
    else:
        raise ValueError(f'Cannot plot type {type(ply)}')


def plot_paths(
    dem: Dem,
    sf: float,
    wb: Sequence[Any],
    fp: Sequence[Any],
    rp: Sequence[Any],
    ww: Sequence[Any],
    border: Polygon,
    cut: Polygon | MultiPolygon,
    cmp: Polygon | MultiPolygon | Any,
    e_poly: Polygon | MultiPolygon,
) -> None:
    """
    Create a map figure to show terrain and selected paths.
    Shapely 2.0-safe: uses .geoms/get_parts instead of iterating geometries.
    """
    fig, ax = plt.subplots()

    # Coordinate transform for raster extent
    x_ll = dem.lon
    y_ll = dem.lat
    corner = np.array([np.min(x_ll), np.min(y_ll)], dtype=float)
    x_m, y_m = cord2dist(x=x_ll, y=y_ll, corner=corner, f=sf)

    # Terrain gradient backdrop
    grad = ndimage.sobel(dem.z)
    ax.imshow(
        grad,
        cmap="pink",
        extent=[x_m.min(), x_m.max(), y_m.min(), y_m.max()],
        origin="lower",
        aspect="equal",
    )

    # Border outline
    draw_polygon(ax, border, fc="none", ec="red", linewidth=0.5)

    # Priority path layers (lists like [cutout, top, support])
    if len(rp) > 0 and rp[1] is not None:
        draw_polygon(ax, rp[1], fc="black")
    if len(fp) > 0 and fp[1] is not None:
        draw_polygon(ax, fp[1], fc="green")
    if len(wb) > 0 and wb[1] is not None:
        draw_polygon(ax, wb[1], fc="teal")
    if len(ww) > 0 and ww[1] is not None:
        draw_polygon(ax, ww[1], fc="blue")

    # CMP (can be Polygon or MultiPolygon)
    if isinstance(cmp, (Polygon, MultiPolygon)):
        draw_polygon(ax, cmp, fc="white", linewidth=0.5)

    # Cut polygons
    draw_polygon(ax, cut, fc="none", linewidth=0.5)

    # e_poly (engrave/emboss outline)
    draw_polygon(ax, e_poly, fc="none", linewidth=1, linestyle="--")

    ax.set_aspect("equal", adjustable="box")
    plt.show()
    plt.pause(0.1)

def lake_elevation(poly: shp.geometry.Polygon, dem: Dem) -> float:
    """
    Calculate the elevation of a lake based on its polygon and DEM data.
    """
    c = np.vstack(poly.exterior.xy).transpose()
    c = dist2cord(c, corner=dem.corner, f=dem.scale_factor)

    z = dem.get_elev(c)  # Perimeter elevation profile
    z = np.sort(z)[int(0.25 * z.shape[0])]  # Use the 25th percentile value as water level
    return z


def generate_stls(
    boundary: list | str = [],
    Rect_Pt: list = [],
    Rect_Pt_rot: list = [],
    rd_include: list = [],
    trail_exclude: list = [],
    trail_include: list = [],
    waterway_include: list = [],
    waterbody: list = [],
    trail_gpx: list = [],
    path_width: float = 0.7,
    support_width: float = 0.9,
    path_clearance: float = 0.1,
    height_factor: float = 2,
    base_height: float = 4,
    min_area: float = 0.5,
    edge_width: float = 1.5,
    max_print_size: list[float] = [248, 198],
    tiles: int = 1,
    dovetails: bool = True,
    dovetail_height: float = 6,
    water_drop: float = 0.5,
    load_area: list = [],
    resolution: int = 30,
    dem_offset: list[float] = [0, 0],
    downsample_factor: int = 1,
    map_only: bool = False,
    compass_loc: list[float] = [],
    compass_size: float = 1.0,
) -> None:
    """

    INPUTS:

    boundary - polygon to define shape of terrain, input as longitude/lattitude array, or gpx file of points
    Rect_Pt - points to bound with rectangular boundary
    Rect_Pt_rot - points to bound with a rotated rectangular boundary
    rd_include - roads names / ids to inlcude, no roads included by default.
    trail_exclude - footpaths names / ids to inlcude, all included by default.
    trail_include - footpaths names / ids to include. If this is not empty, all other trails are exluded and "trail_exclude" is ignored
    waterway_include - water paths to inlcude, not inlcuding polygon water bodies
    waterway_include - water areas to include (lakes), current set to print at one level, just below surrounding land.  Will not work well for rivers that are not flat.
    trail_gpx - gpx file to use for trail path
    path_width - path (road, footpaths and waterway) print top width, 3 total passes works best.
    support_width - width of base (one pass less than top)
    path_clearance - clearance between path prints and cutouts.
    height_factor - exaggeration factor for elevation. 1.0 makes on same scale as horizontal dimensions.
    base_height - minium print thickness, should be > 9 if tiling is used.
    min_area - min area for any printed shape
    edge_width - width of terrian border with no path cutouts
    max_print_size - maximum print dimension.  will scale and rotate print to fit this.
    tiles - number of tiles to use for terrain print 
    dovetails - boolean to set if tiles are joined with dovetail inserts
    dovetail_height - Thickness of generated dovetail inserts and cutouts, 20 max
    water_drop - water ways printed slightly lower than other paths and terrain
    load_area - overide automatic area selection
    resolution - 10 or 30, for 10/30 meter resolution DEM.
    dem_offset - offset DEM relative to OSM data to account for shifts in data.
    downsample_factor, integer factor to reduce resolution of terrain surface, needed for larger models.  1mm resolution is reasonable minimum.
    map_only - stop after generating map to review (no boolean ops), recomend to do this first until all desired features look corrects oi it doesn't get hung up on boolean operations for hours.
    compass_loc - XY location for printed compass.
    compass_size - scale factor for size of compass (Letter N might reqiure 0.25 nozzle)
    """
    if not os.path.isdir('print_files/'):
        os.mkdir('print_files/')
    if not os.path.isdir('temp/'):
        os.mkdir('temp/')

    # clear temp files
    fileList = glob.glob('temp/*.stl')
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
            
            
    # clear print files
    fileList = glob.glob('print_files/*.stl')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    
    #pull print shape from gpx file if string is input
    if isinstance(boundary,str):
        gpx_file = open(boundary, 'r')

        gpx = gpxpy.parse(gpx_file)
        poly=[]
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    poly.append([point.latitude, point.longitude])
        boundary=np.array(poly)
    else:
        boundary=np.array(boundary)
    if len(boundary)==0 and len(Rect_Pt)>1:
        Rect_Pt=np.array(Rect_Pt)
        Rect_Pt=np.vstack((Rect_Pt.min(0),[Rect_Pt[:,0].min(),Rect_Pt[:,1].max()],Rect_Pt.max(0),[Rect_Pt[:,0].max(),Rect_Pt[:,1].min()]))
        boundary=Rect_Pt
    if len(boundary)==0 and len(Rect_Pt_rot)>2:
        Rect_Pt_rot=np.flip(Rect_Pt_rot,axis=1)
        c=Rect_Pt_rot.min(axis=0)
        Rect_Pt_rot=shp.geometry.Polygon(cord2dist(xy=Rect_Pt_rot,corner=c))
        boundary=Rect_Pt_rot.minimum_rotated_rectangle
        boundary=np.flip(dist2cord(xy=np.array(boundary.exterior.xy).transpose(),corner=c),axis=1)

    if not any(load_area): #load area bounds polygon unless larger area is input
        load_area=np.vstack((np.min(boundary,axis=0),np.max(boundary,axis=0)))
        
    boundary=np.flip(boundary,axis=1) #from here on in, we use x,y notation
    dem=Dem(boundary,resolution,dem_offset,downsample_factor)
    
    #dem.plot_elev()

    #find the largest print that fits (rotated) within print size a return scale factor and x/y=0 corner of print in coords'
    if tiles>1:
        if not dovetails:
            dovetail_height=0
        scale_factor,corner,edge_poly,cutouts=print_scaling_tiled(dem,boundary,max_print_size,tiles,dovetail_height)
    else:
        scale_factor,corner,edge_poly=print_scaling(dem,boundary,max_print_size)
        cutouts=[]

    # offsets=[support_width/2,path_width/2,(path_width+path_clearance)/2]
    offsets=[(path_width+path_clearance)/2]
    
    
    print('Processing OSM results')
    # start = time.time()

    
    areaStr=str("%f, %f, %f, %f" % (np.min(boundary[:,1]),np.min(boundary[:,0]),np.max(boundary[:,1]),np.max(boundary[:,0])))
    num_attempt=0
    OSMresults=-1
    while not str(OSMresults.__class__)=="<class 'overpy.Result'>" and num_attempt<=5:
        try:
            #roads and railways
            #result = api.query("way(" + areaStr + ") [""highway""];   (._;>;); out body;")
            # result = api.query("way(" + areaStr + ") [""railway""];   (._;>;); out body;")
            OSMresults = api.query("(rel(" + areaStr + ")[""route""];way(" + areaStr + ") [""highway""];way(" + areaStr + ") [""railway""];way(" + areaStr + ") [""waterway""];rel(" + areaStr + ") [""water""];way(" + areaStr + ") [""water""];);   (._;>;); out body;")
        except:
            num_attempt=num_attempt+1
            print('|')
            time.sleep(5)


    Footpaths=get_footpaths(OSMresults,trail_exclude,trail_include,trail_gpx,corner,scale_factor,offsets,base_height)
    
    Roads=get_roads(OSMresults,rd_include,corner,scale_factor,offsets,base_height)
    
    Waterways=get_waterways(OSMresults,waterway_include,corner,scale_factor,offsets,base_height,map_only)
    Waterbodies=get_waterbodies(OSMresults,waterbody,corner,scale_factor,path_clearance,base_height,height_factor,dem)
    
    boundary=shp.geometry.Polygon(cord2dist(xy=boundary,corner=corner,f=scale_factor))

    borders=offset_polygon(boundary,[-edge_width-path_clearance,-edge_width-path_clearance/2,-edge_width,0])
    print('2D Boolean Operations')
    
        
    Roads,Footpaths,Waterbodies,Waterways,Cutout=binary_operations([Roads,Footpaths,Waterbodies,Waterways],borders,path_width,support_width,path_clearance,min_area)

    if len(compass_loc)==2: #redundant with other compass section
        print('generating compass')
        resource_path = resources.files(__package__).joinpath("resources")
        cmp=tm.load(resource_path.joinpath("Compass.stl"))
        t=np.eye(4)
        t[:2, :2] *= compass_size
        cmp.apply_transform(t) #scale
        
        cmp.apply_translation([compass_loc[0], compass_loc[1],0])
      
        c_poly = cmp.section(plane_origin=[0,0,-5],plane_normal=[0,0,1])

        p=[]
        for e in c_poly.entities:
            p.append(shp.geometry.Polygon(c_poly.vertices[e.points,:]))
        if len(p)>1:
            c_poly=shp.geometry.MultiPolygon(p)
        else:
            c_poly=shp.geometry.Polygon(p[0])
    else:
        c_poly=[]
    
    if not map_only:
        if len(compass_loc)==2:
           #get elevation around edge ofcompass polygon
           edge=[]
           for p in c_poly.geoms:
               edge.append(np.array(p.exterior.xy).T)
           edge=dist2cord(np.vstack(edge),corner=corner,f=scale_factor)

           z=dem.get_elev(edge)
           z=z-np.min(dem.z)
           z=z*scale_factor*height_factor+base_height+1

           
           #Cut off bottom to just below min elevation
           box=tm.creation.box([200,200,200])
           box.apply_translation([compass_loc[0], compass_loc[1],min(z)-2-1-100])
           
           cmp.apply_translation([0,0,max(z)])
           cmp.export('temp/c1.stl')
           box.export('temp/b1.stl')
           cmp=tm.boolean.difference([cmp,box])
           cmp.export('print_files/compass.stl')
           
           
           cmp_cut=tm.load(resource_path.joinpath("Compass_cutout.stl"))
           cmp_cut.apply_transform(t)
           cmp_cut.apply_translation([0,0,min(z)+50-2])
           box=tm.creation.box([200,200,200])
           cmp_cut.apply_translation([compass_loc[0], compass_loc[1],0])
           cmp_cut=[cmp_cut]
        else:
            cmp_cut=[]
        wb_heights2=[]
        if len(Waterbodies)>0:
            grid=np.meshgrid(dem.lat,dem.lon)
            x,y=cord2dist(x=grid[1].flatten(),y=grid[0].flatten(),corner=corner,f=scale_factor)
            
            if Waterbodies[1].geom_type == 'MultiPolygon':
                for i in range(len(Waterbodies[1].geoms)):
    
                    z=lake_elevation(Waterbodies[1].geoms[i],dem)
                    wb_heights2.append((z-np.min(dem.z))*scale_factor*height_factor+1-3+base_height) #+1 to line up terrain
                    idx=included_points(Waterbodies[1].geoms[i],dem,corner,scale_factor)             
                    dem.z[idx]=z #set points to nominal elevation.     
            else:
                z=lake_elevation(Waterbodies[1],dem)
                wb_heights2.append((z-np.min(dem.z))*scale_factor*height_factor+1-3+base_height) #+1 to line up terrain
                idx=included_points(Waterbodies[1],dem,corner,scale_factor)             
                dem.z[idx]=z #set points to nominal elevation. 
  
        print('Meshing')
        rd_msh=meshgen2(Roads,dem,corner,scale_factor,height_factor,base_height,fname='temp/road')
        fp_msh=meshgen2(Footpaths,dem,corner,scale_factor,height_factor,base_height,fname='temp/trail')
        ww_msh=meshgen2(Waterways,dem,corner,scale_factor,height_factor,base_height,fname='temp/water')
        wb_msh=meshgen_wb(Waterbodies,[20,1.5,3],'temp/waterbodies',wb_heights2)


        if edge_poly.geom_type == 'MultiPolygon':
            terrain=[]
            for e in edge_poly.geoms:
                b=e.intersection(boundary)
                if b.geom_type!='GeometryCollection':
                    t=terrain_mesh(dem,b,scale_factor,corner,height_factor,base_height,water_drop)
                    terrain.append(t)
        else:
            terrain=[terrain_mesh(dem,boundary,scale_factor,corner,height_factor,base_height,water_drop)]
            
        terrain_all=terrain_mesh(dem,boundary,scale_factor,corner,height_factor,base_height,water_drop)
        terrain_all.export('print_files/terrain.stl')
       
        
        print('Cutting Terrain Model')
        
        cutlist=[]
        i=1
        
        
        for m in [cmp_cut,ww_msh,wb_msh,rd_msh,fp_msh,cutouts]:
            
            if len(m)>0:
                cutlist.append(m[0])
                m[0].export('temp/cc'+str(i)+'.stl')
                # print(m[0].is_watertight)
                i=i+1
        for i in range(len(terrain)):
            terrain[i].export('temp/t-'+str(i+1)+'.stl')
            if not terrain[i].is_watertight:
                success = terrain[i].fill_holes()
                if not success:
                    print(f'Failed to make a watertight volume from terrain segment {i}')
            #tx=terrain[i]
            #for j in range(len(cutlist)):
            #    tx=tm.boolean.difference([tx,cutlist[j]], engine='manifold')
            #    print(tx.is_watertight)
            #    if not tx.is_watertight:
            #        tx.fill_holes()
            #        print(tx.is_watertight)
            terrain[i]=tm.boolean.difference([terrain[i]]+cutlist)
            terrain[i].export('print_files/terrain-'+str(i+1)+'.stl')

        print('Building Inserts')
        #cut top of top section
        
        fp_msh[1]=tm.boolean.intersection([fp_msh[1],terrain_all])
        
        if len(rd_msh)>0:
            rd_msh[1]=tm.boolean.intersection([rd_msh[1],terrain_all])
        if len(ww_msh)>0:
            ww_msh[1]=tm.boolean.intersection([ww_msh[1],terrain_all])

         #cut top of support section
        terrain_all.apply_translation([0,0,-1])
        
        fp_msh[2]=tm.boolean.intersection([fp_msh[2],terrain_all])
        
        if len(rd_msh)>0:
            rd_msh[2]=tm.boolean.intersection([rd_msh[2],terrain_all])
        if len(ww_msh)>0:
            ww_msh[2]=tm.boolean.intersection([ww_msh[2],terrain_all])

        
        terrain_all.apply_translation([0,0,-1])
             
        # cut off bottom of top section, and join with support section
        fp_msh[1]=tm.boolean.difference([fp_msh[1],terrain_all])
        fp_msh[2]=tm.boolean.union([fp_msh[1],fp_msh[2]])
        fp_msh[2].export('print_files/footpaths.stl')
        
        if len(rd_msh)>0:
            rd_msh[1]=tm.boolean.difference([rd_msh[1],terrain_all])
            rd_msh[2]=tm.boolean.union([rd_msh[1],rd_msh[2]])
            rd_msh[2].export('print_files/roads.stl')

            
        if len(ww_msh)>0:
            ww_msh[1]=tm.boolean.difference([ww_msh[1],terrain_all])
            ww_msh[2]=tm.boolean.union([ww_msh[1],ww_msh[2]])
            ww_msh[2].export('print_files/waterways.stl')

        
        if len(wb_msh)>0:
            wb_msh[1].apply_translation([0,0,1.5])
            wb_msh[1]=tm.boolean.union([wb_msh[1],wb_msh[2]])
            wb_msh[1].export('print_files/waterbodies.stl')

        
    plot_paths(dem,scale_factor,Waterbodies,Footpaths,Roads,Waterways,boundary,Cutout,c_poly,edge_poly)
    


    print("COMPLETE")
