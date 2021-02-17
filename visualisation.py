import cartopy
import cartopy.crs as ccrs
import fiona
import geopandas
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import pandas
import rasterio
import rasterio.mask
import rasterio.plot

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import shape

import gc
import glob
import json
import os

matplotlib.rcParams["figure.figsize"] = (12, 13)
matplotlib.rcParams["figure.dpi"] = 72
matplotlib.rcParams["font.size"] = 16

# read shared data folder location from config
with open("config.json", 'r') as fh:
    base_folder = json.load(fh)['data_folder']

# read Arc LADs
# lads_path = os.path.join(base_folder, 'GIS Data', 'arc_lad_uk16.gpkg')
# lads_df = geopandas.read_file(lads_path)

# lad_centroids_df = lads_df[lads_df.in_arc == 1].copy()
# lad_centroids_df["geometry"] = lad_centroids_df.geometry.centroid
# lad_centroids_df = lad_centroids_df[["desc", "geometry"]].rename(columns={"desc":"label"})
# lad_centroids_df.to_file(
#     os.path.join(base_folder, 'Visual Narrative', 'Data', 'arc_lad_centroids.geojson'), 
#     driver="GeoJSON"
# )
# lad_centroids_df.head()

# read Arc MSOAs
# msoa_path = os.path.join(base_folder, 'GIS Data', 'msoa_arc.gpkg')
# msoa_df = geopandas.read_file(msoa_path)

# msoa_names = pandas.read_csv(
#     os.path.join(base_folder, 'Visual Narrative', 'Data', 'MSOA-Names-1.5.0.csv')
# )[["msoa11cd", "msoa11hclnm"]]
# msoa_names.head(1)

# msoa_centroids_df = msoa_df.copy()
# msoa_centroids_df["geometry"] = msoa_centroids_df.geometry.centroid
# msoa_centroids_df = msoa_centroids_df[["msoa11cd", "geometry"]].set_index("msoa11cd")
# msoa_centroids_df = msoa_centroids_df \
#     .join(msoa_names.set_index("msoa11cd"), how="left") \
#     .rename(columns={'msoa11hclnm': 'label'})
# msoa_centroids_df.to_file(
#     os.path.join(base_folder, 'Visual Narrative', 'Data', 'arc_msoa_centroids.geojson'), 
#     driver="GeoJSON"
# )
# msoa_centroids_df.head(1)

outline_path = os.path.join(
    base_folder, 'Visual Narrative', 'Data', 'Arc Outline', 'arc-outline.gpkg')
with fiona.open(outline_path, "r") as shapefile:
    arc_mask = [feature["geometry"] for feature in shapefile]
    outline = [shape(p) for p in arc_mask]

arc_extent = (418_000, 573_000, 170_000, 325_000)
cty_extent = (475_000, 500_000, 225_000, 250_000)
ngb_extent = (487_000, 492_000, 231_500, 236_500)
extents = {
    'arc': arc_extent,
    'cty': cty_extent,
    'ngb': ngb_extent
}

green = '#219653' 
blue = '#2D9CDB'
yellow = '#F2C94C'
red = '#EB5757'
grey = '#DDDDDD'

# Natural Capital geodatabase location
nc_paths = glob.glob(
    os.path.join(base_folder, 'Scenarios', 'Natural Capital', 'Arc_*.tif'))
nc_paths

def files_raster_min_max(paths, mask=None):
    # loop over to get vmax
    vmin, vmax = 0, 0
    for fname in paths:
        with rasterio.open(fname) as ds:
            if mask is not None:                
                data, _ = rasterio.mask.mask(ds, mask, crop=True)
                data = data[0]
            else:
                data = ds.read(1)
            data_max = numpy.max(data)
            data_min = numpy.min(data)
            if data_max > vmax:
                vmax = data_max
            if data_min < vmin:
                vmin = data_min
    
    if vmin < 0:
        vmin = 0
        
    return vmin, vmax

def plot_map(raster, raster_extent, extent, label=None, cmap='Greens', norm=None):
    osgb = ccrs.epsg(27700)
    fig, ax = plt.subplots(subplot_kw={'projection':osgb}, figsize=(12, 13))
    ax.set_frame_on(False)
    ax = plt.axes([0, 0.07, 1, 1], projection=osgb)
    ax.set_extent(extent, crs=osgb)
    ax.set_frame_on(False) # don't draw axes outline/background rectangle

    # add the image
    ax.imshow(raster, origin='upper', extent=raster_extent, transform=osgb, cmap=cmap, norm=norm)
    
    # add the colorbar
    cax = inset_axes(
        ax,
        width="40%",  
        height="3%",
        loc='lower left',
        bbox_to_anchor=(0.05, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=label, orientation='horizontal')
    return ax

# hard code 0-10 range - note that habitat scores go above 10, simply capping here
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

def do_plot(data, data_extent, fname, extents, norm):    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", green])

    service = os.path.basename(fname) \
        .replace('Arc_FreeData_', '') \
        .replace('_25m_MCA.tif', '')   
    
    text = service.replace('Q', ' Q') \
        .replace('Control', ' Control') \
        .replace('Carbon', 'Carbon Storage') \
        .replace('Erosion', 'Erosion Prevention') \
        .replace('Flood', 'Flood Protection') \
        .replace('Food', 'Food Production') \
        .replace('Noise', 'Noise Reduction')
    
    label = f"{text} Score"
    print(label)

    for zoom in ('arc', 'cty', 'ngb'):
        ax = plot_map(data, data_extent, extents[zoom], cmap=cmap, norm=norm, label=label)
        plt.savefig(f"natcap_{service.lower()}_{zoom}.png")
        
        fig = plt.gcf()
        fig.clf()
        plt.close()
        gc.collect()
    

for fname in nc_paths:
    with rasterio.open(fname) as ds:
        data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        data = data[0]  # ignore first dimension, just want 2D array
        data_extent = rasterio.plot.plotting_extent(ds)

    if 'Habitat' in fname:
        m = numpy.max(data)
        print("Habitat max", m)
        data = (data.astype("float") * 10) / m
        
    do_plot(data, data_extent, fname, extents, norm)
        
    del data    
    gc.collect() 

def get_food_non(nc_paths, arc_mask):
    non_food = None
    food = None
    for fname in nc_paths:
        with rasterio.open(fname) as ds:
            data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
            data = data[0]  # ignore first dimension, just want 2D array
            data_extent = rasterio.plot.plotting_extent(ds)
        if 'Food' in fname:
            food = data
            continue
            
        if 'Habitat' in fname:
            m = numpy.max(data)
            print("Habitat max", m)
            data = (data * 10) / m
            
        if non_food is None:
            non_food = data
        else:
            non_food = numpy.maximum.reduce([non_food, data])
            
        del data
        gc.collect
    food[food < non_food] = -1
    non_food[non_food < food] = -1
    return food, non_food
food, non_food = get_food_non(nc_paths, arc_mask)

with rasterio.open(nc_paths[0]) as ds:
    data_extent = rasterio.plot.plotting_extent(ds)
    
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
greens = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff00", green])
reds = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff00", yellow])
greens.set_under(color=(1, 1, 1, 0))
reds.set_under(color=(1, 1, 1, 0))

for zoom in ('arc', 'cty', 'ngb'):    
    osgb = ccrs.epsg(27700)
    fig, ax = plt.subplots(subplot_kw={'projection':osgb}, figsize=(12, 13))
    ax.set_frame_on(False)
    ax = plt.axes([0, 0.07, 1, 1], projection=osgb)
    ax.set_extent(extents[zoom], crs=osgb)
    
    ax.imshow(food, origin='upper', extent=data_extent, transform=osgb, cmap=reds, norm=norm)
    ax.imshow(non_food, origin='upper', extent=data_extent, transform=osgb, cmap=greens, norm=norm)
    
    # add the colorbars
    nf_cax = inset_axes(
        ax,
        width="40%",  
        height="3%",
        loc='lower left',
        bbox_to_anchor=(0.05, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=greens), cax=nf_cax, 
                 orientation='horizontal',
                 label="Non-food Natural Capital Score")
    
    f_cax = inset_axes(
        ax,
        width="40%",  
        height="3%",
        loc='lower left',
        bbox_to_anchor=(0.55, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=reds), cax=f_cax, 
                 orientation='horizontal',
                 label="Food Natural Capital Score")
    
    # don't draw axes outline/background rectangle
    ax.set_frame_on(False) 
    f_cax.set_frame_on(False)
    nf_cax.set_frame_on(False)
    plt.savefig(f"natcap_combined_{zoom}.png")

    fig = plt.gcf()
    fig.clf()
    plt.close()
    gc.collect()

density_paths = glob.glob(
    os.path.join(base_folder, 'Scenarios', 'UDM', 'ATI FINAL', 'Inputs', 'Density Surfaces', 'TIFF', '*.tif'))
density_paths

vmin, vmax = files_raster_min_max(density_paths, arc_mask)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

print("Range:", vmin, vmax)

for fname in density_paths:
    with rasterio.open(fname) as ds:
        data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        data = data[0]  # ignore first dimension, just want 2D array
        data_extent = rasterio.plot.plotting_extent(ds)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", blue])

    dwellings, policy, _, _ = os.path.basename(fname).lower().split('_')
    if dwellings == 'expansion':
        dwellings = 'exp'
    elif dwellings == 'settlements':
        dwellings = 'set'
    else:
        assert False, dwellings

    for zoom in ('arc', 'cty', 'ngb'):
        _ = plot_map(data, data_extent, extents[zoom], cmap=cmap, norm=norm, label="Potential dwellings per hectare")
        plt.savefig(f"density_{policy}_{dwellings}_{zoom}.png")
        plt.close()

    print(fname)

attractor_paths = glob.glob(
    os.path.join(base_folder, 'Scenarios', 'UDM', 'ATI FINAL', 'Inputs', 'Attractors', 'TIFF', '*.tif'))
attractor_paths

def titleify(str_, sep=" "):
    words = iter(str_.split(sep))
    stop = {"by", "the", "of", "to"}
    
    label = next(words).title() + " "
    for word in words:
        if word not in stop:
            label += word.title()
        else:
            label += word
        label += " "
    return label.strip()
titleify("the intro to the software")

vmin, vmax = files_raster_min_max(attractor_paths, arc_mask)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

print("Range:", vmin, vmax)


for fname in attractor_paths:
    with rasterio.open(fname) as ds:
        data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        data = data[0]  # ignore first dimension, just want 2D array
        data_extent = rasterio.plot.plotting_extent(ds)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", yellow])
    slug = os.path.basename(fname).replace('.tif', '')
    label = titleify(slug, sep='_')

    for zoom in ('arc', 'cty', 'ngb'):
        _ = plot_map(data, data_extent, extents[zoom], cmap=cmap, norm=norm, label=label)
        plt.savefig(f"attractor_{slug}_{zoom}.png")
        plt.close()

    print(slug, label)

constraint_paths = glob.glob(
    os.path.join(base_folder, 'Scenarios', 'UDM', 'ATI FINAL', 'Inputs', 'Constraints', 'TIFF', '*.tif'))
constraint_paths

vmin, vmax = files_raster_min_max(constraint_paths, arc_mask)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

print("Range:", vmin, vmax)

for fname in constraint_paths:
    with rasterio.open(fname) as ds:
        data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        data = data[0]  # ignore first dimension, just want 2D array
        data_extent = rasterio.plot.plotting_extent(ds)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", grey])
    dwellings, grey_green, _ = os.path.basename(fname).split('_')
    
    if dwellings == 'expansion':
        dwellings = 'exp'
    elif dwellings == 'settlements':
        dwellings = 'set'
    else:
        assert False, dwellings

    for zoom in ('arc', 'cty', 'ngb'):
        _ = plot_map(data, data_extent, extents[zoom], cmap=cmap, norm=norm, label="Combined Constraints")
        plt.savefig(f"constraint_{grey_green}_{dwellings}_{zoom}.png")
        plt.close()

    print(fname)

suitability_paths = glob.glob(
    os.path.join(base_folder, 'Scenarios', 'UDM', 'ATI FINAL', 'Outputs', 'Suitability Surfaces', 'TIFF', '*.tif'))
suitability_paths

vmin, vmax = files_raster_min_max(suitability_paths, arc_mask)
if vmin < 0:
    vmin = 0
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

print("Range:", vmin, vmax)
if vmin < 0:
    vmin = 0

for fname in suitability_paths:
    with rasterio.open(fname) as ds:
        data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        data = data[0]  # ignore first dimension, just want 2D array
        data_extent = rasterio.plot.plotting_extent(ds)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", yellow])
    dwellings, grey_green, _ = os.path.basename(fname).split('_')
    
    if dwellings == 'expansion':
        dwellings = 'exp'
    elif dwellings == 'settlements':
        dwellings = 'set'
    else:
        assert False, dwellings

    for zoom in ('arc', 'cty', 'ngb'):
        _ = plot_map(data, data_extent, extents[zoom], cmap=cmap, norm=norm, label="Combined Suitability Score")
        plt.savefig(f"suit_{grey_green}_{dwellings}_{zoom}.png")
        plt.close()

    print(fname)

dwellings_paths = sorted(glob.glob(
    os.path.join(base_folder, 'Scenarios', 'UDM', 'ATI FINAL', 'Outputs', '**', 'Dwellings', 'TIFF', '*.tif')))
dwellings_paths

development_paths = sorted(glob.glob(
    os.path.join(base_folder, 'Scenarios', 'UDM', 'ATI FINAL', 'Outputs', '**', 'Development', 'TIFF', '*.tif')))
development_paths

vmin, vmax = files_raster_min_max(dwellings_paths)
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
vmin, vmax

def dev_plot(fname, dev_fname):
    with rasterio.open(fname) as ds:
        data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        data = data[0]  # ignore first dimension, just want 2D array
        # mask out zero values
        data[data == 0] = -1
        data_extent = rasterio.plot.plotting_extent(ds)
    
    with rasterio.open(dev_fname) as ds:
        dev_data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        dev_data = dev_data[0]  # ignore first dimension, just want 2D array
        # mask out all cells with value 2 (new development)
        dev_data[dev_data == 2] = -2
        # bump 0 and 1 (undeveloped and previously developed) to hack colour
        dev_data[dev_data == 1] = 5
        dev_data[dev_data == 0] = 1
        
    dev_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

    year, rate, dwellings, policy, _ = os.path.basename(fname).split('_')
    
    if dwellings == 'expansion':
        dwellings = 'exp'
    elif dwellings == 'settlements':
        dwellings = 'set'
    else:
        assert False, dwellings
        
    out_name = f"dwellings_{policy}_{dwellings}_{rate}_{year}_zoom.png"
        
    reds = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffadad", red])
    blues = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", blue])
    reds.set_under(color=(1, 1, 1, 0))
    blues.set_under(color=(1, 1, 1, 0))

    for zoom in ('arc', 'cty', 'ngb'):        
        osgb = ccrs.epsg(27700)
        fig, ax = plt.subplots(subplot_kw={'projection':osgb}, figsize=(12, 13))
        ax.set_frame_on(False)
        ax = plt.axes([0, 0.07, 1, 1], projection=osgb)
        ax.set_frame_on(False)
        ax.set_extent(extents[zoom], crs=osgb)

        ax.imshow(dev_data, origin='upper', extent=data_extent, transform=osgb, cmap=blues, norm=dev_norm)
        ax.imshow(data, origin='upper', extent=data_extent, transform=osgb, cmap=reds, norm=norm)

        # add the colorbar
        cax = inset_axes(
            ax,
            width="40%",  
            height="3%",
            loc='lower left',
            bbox_to_anchor=(0.05, -0.05, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=reds), cax=cax, 
                     orientation='horizontal',
                     label="New Dwellings (per hectare)")
        
        # add extra legend bits
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='No dwellings', markerfacecolor='#dff0f9', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Current development', markerfacecolor='#95cdec', markersize=10),
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.55, -0.075), loc='lower left', borderaxespad=0.)
        
        plt.savefig(out_name.replace('zoom', zoom))
        
        fig = plt.gcf()
        fig.clf()
        plt.close()

for fname, dev_fname in zip(dwellings_paths, development_paths):
    print(fname)
    print(dev_fname)
    dev_plot(fname, dev_fname)
    gc.collect()

