import cartopy.crs as ccrs
import fiona
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy
import rasterio
import rasterio.mask
import rasterio.plot

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import json
import os

matplotlib.rcParams["figure.figsize"] = (12, 13)
matplotlib.rcParams["figure.dpi"] = 72
matplotlib.rcParams["font.size"] = 16

#
# Constants for plotting
#
EXTENTS = {
    'arc': (418_000, 573_000, 170_000, 325_000),
    'cty': (475_000, 500_000, 225_000, 250_000),
    'ngb': (487_000, 492_000, 231_500, 236_500)
}
GREEN = '#219653'
BLUE = '#2D9CDB'
YELLOW = '#F2C94C'
RED = '#EB5757'
GREY = '#DDDDDD'


def main(base_folder):
    """Plot images from OpenUDM outputs
    """
    suitability_path = os.path.join(base_folder, 'out_cell_suit.asc')
    dwellings_path = os.path.join(base_folder, 'out_cell_dph.asc')
    development_path = os.path.join(base_folder, 'out_cell_dev.asc')

    outline_path = os.path.join(base_folder, 'arc-outline.gpkg')
    with fiona.open(outline_path, "r") as shapefile:
        arc_mask = [feature["geometry"] for feature in shapefile]

    #
    # Suitability
    #
    vmin, vmax = file_raster_min_max(suitability_path, arc_mask)
    if vmin < 0:
        vmin = 0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    with rasterio.open(suitability_path) as ds:
        data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
        data = data[0]  # ignore first dimension, just want 2D array
        data_extent = rasterio.plot.plotting_extent(ds)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", YELLOW])

    for zoom in ('arc', 'cty', 'ngb'):
        plot_map(data, data_extent, EXTENTS[zoom], cmap=cmap, norm=norm, label="Combined Suitability Score")
        plt.savefig(f"suitability_{zoom}.png")
        plt.close()

    #
    # Development and dwellings
    #
    vmin, vmax = file_raster_min_max(dwellings_path)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    plot_development_and_dwellings(dwellings_path, development_path, norm, arc_mask)


def files_raster_min_max(paths, mask=None):
    # loop over to get vmax
    vmin, vmax = 0, 0
    for fname in paths:
        vmin, vmax = file_raster_min_max(fname, mask, vmin, vmax)
    return vmin, vmax


def file_raster_min_max(fname, mask=None, vmin=0, vmax=0):
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


def plot_development_and_dwellings(fname, dev_fname, norm, arc_mask):
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

    reds = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffadad", RED])
    blues = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", BLUE])
    reds.set_under(color=(1, 1, 1, 0))
    blues.set_under(color=(1, 1, 1, 0))

    for zoom in ('arc', 'cty', 'ngb'):
        osgb = ccrs.epsg(27700)
        fig, ax = plt.subplots(subplot_kw={'projection':osgb}, figsize=(12, 13))
        ax.set_frame_on(False)
        ax = plt.axes([0, 0.07, 1, 1], projection=osgb)
        ax.set_frame_on(False)
        ax.set_extent(EXTENTS[zoom], crs=osgb)

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

        plt.savefig(f"dwellings_{zoom}.png")

        fig = plt.gcf()
        fig.clf()
        plt.close()


if __name__ == '__main__':
    with open("config.json", 'r') as fh:
        base_folder = json.load(fh)['data_folder']
    main(base_folder)
