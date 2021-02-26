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
import gc
import glob
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
    outline_path = os.path.join(base_folder, 'arc-outline.gpkg')

    # TODO check intended folder structure
    nc_paths = glob.glob(
        os.path.join(base_folder, 'natural_capital', '*.tif'))
    density_paths = glob.glob(
        os.path.join(base_folder, 'inputs', 'Density Surfaces', 'TIFF', '*.tif'))
    attractor_paths = glob.glob(
        os.path.join(base_folder, 'inputs', 'Attractors', 'TIFF', '*.tif'))
    constraint_paths = glob.glob(
        os.path.join(base_folder, 'inputs', 'Constraints', 'TIFF', '*.tif'))
    suitability_paths = glob.glob(
        os.path.join(base_folder, 'outputs', 'Suitability Surfaces', 'TIFF', '*.tif'))
    dwellings_paths = sorted(glob.glob(
        os.path.join(base_folder, 'outputs', '**', 'Dwellings', 'TIFF', '*.tif')))
    development_paths = sorted(glob.glob(
        os.path.join(base_folder, 'outputs', '**', 'Development', 'TIFF', '*.tif')))

    with fiona.open(outline_path, "r") as shapefile:
        arc_mask = [feature["geometry"] for feature in shapefile]

    #
    # Natural Capital scores
    #

    # hard code 0-10 range - note that habitat scores go above 10, simply capping here
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

    for fname in nc_paths:
        with rasterio.open(fname) as ds:
            data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
            data = data[0]  # ignore first dimension, just want 2D array
            data_extent = rasterio.plot.plotting_extent(ds)

        if 'Habitat' in fname:
            m = numpy.max(data)
            data = (data.astype("float") * 10) / m

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

        plot_natural_capital_score(data, data_extent, service, label, norm)

        del data
        gc.collect()

    #
    # Natural Capital food/non-food
    #
    food, non_food = get_food_non(nc_paths, arc_mask)

    with rasterio.open(nc_paths[0]) as ds:
        data_extent = rasterio.plot.plotting_extent(ds)

    plot_food_non_food_natural_capital(food, non_food, data_extent)

    #
    # Density
    #
    vmin, vmax = files_raster_min_max(density_paths, arc_mask)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for fname in density_paths:
        with rasterio.open(fname) as ds:
            data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
            data = data[0]  # ignore first dimension, just want 2D array
            data_extent = rasterio.plot.plotting_extent(ds)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", BLUE])

        # TODO check filename pattern
        dwellings, policy, _, _ = os.path.basename(fname).lower().split('_')
        if dwellings == 'expansion':
            dwellings = 'exp'
        elif dwellings == 'settlements':
            dwellings = 'set'
        else:
            assert False, dwellings

        for zoom in ('arc', 'cty', 'ngb'):
            _ = plot_map(data, data_extent, EXTENTS[zoom], cmap=cmap, norm=norm, label="Potential dwellings per hectare")
            plt.savefig(f"density_{policy}_{dwellings}_{zoom}.png")
            plt.close()

    #
    # Attractors
    #
    vmin, vmax = files_raster_min_max(attractor_paths, arc_mask)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for fname in attractor_paths:
        with rasterio.open(fname) as ds:
            data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
            data = data[0]  # ignore first dimension, just want 2D array
            data_extent = rasterio.plot.plotting_extent(ds)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", YELLOW])

        # TODO split any extension
        slug = os.path.basename(fname).replace('.tif', '')
        label = titleify(slug, sep='_')

        for zoom in ('arc', 'cty', 'ngb'):
            _ = plot_map(data, data_extent, EXTENTS[zoom], cmap=cmap, norm=norm, label=label)
            plt.savefig(f"attractor_{slug}_{zoom}.png")
            plt.close()

    #
    # Constraints
    #
    vmin, vmax = files_raster_min_max(constraint_paths, arc_mask)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for fname in constraint_paths:
        with rasterio.open(fname) as ds:
            data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
            data = data[0]  # ignore first dimension, just want 2D array
            data_extent = rasterio.plot.plotting_extent(ds)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", GREY])

        # TODO check filename pattern
        dwellings, grey_green, _ = os.path.basename(fname).split('_')

        if dwellings == 'expansion':
            dwellings = 'exp'
        elif dwellings == 'settlements':
            dwellings = 'set'
        else:
            assert False, dwellings

        for zoom in ('arc', 'cty', 'ngb'):
            _ = plot_map(data, data_extent, EXTENTS[zoom], cmap=cmap, norm=norm, label="Combined Constraints")
            plt.savefig(f"constraint_{grey_green}_{dwellings}_{zoom}.png")
            plt.close()

    #
    # Suitability
    #
    vmin, vmax = files_raster_min_max(suitability_paths, arc_mask)
    if vmin < 0:
        vmin = 0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for fname in suitability_paths:
        with rasterio.open(fname) as ds:
            data, _ = rasterio.mask.mask(ds, arc_mask, crop=True)
            data = data[0]  # ignore first dimension, just want 2D array
            data_extent = rasterio.plot.plotting_extent(ds)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", YELLOW])

        # TODO check filename pattern
        dwellings, grey_green, _ = os.path.basename(fname).split('_')

        if dwellings == 'expansion':
            dwellings = 'exp'
        elif dwellings == 'settlements':
            dwellings = 'set'
        else:
            assert False, dwellings

        for zoom in ('arc', 'cty', 'ngb'):
            plot_map(data, data_extent, EXTENTS[zoom], cmap=cmap, norm=norm, label="Combined Suitability Score")
            plt.savefig(f"suit_{grey_green}_{dwellings}_{zoom}.png")
            plt.close()

    #
    # Development and dwellings
    #
    vmin, vmax = files_raster_min_max(dwellings_paths)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    vmin, vmax

    for fname, dev_fname in zip(dwellings_paths, development_paths):
        dev_plot(fname, dev_fname, norm, arc_mask)
        gc.collect()


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


def plot_natural_capital_score(data, data_extent, service, label, norm):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", GREEN])

    for zoom in ('arc', 'cty', 'ngb'):
        plot_map(data, data_extent, EXTENTS[zoom], cmap=cmap, norm=norm, label=label)
        plt.savefig(f"natcap_{service.lower()}_{zoom}.png")

        fig = plt.gcf()
        fig.clf()
        plt.close()
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

def plot_food_non_food_natural_capital(food, non_food, data_extent):
    norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    greens = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff00", GREEN])
    reds = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff00", YELLOW])
    greens.set_under(color=(1, 1, 1, 0))
    reds.set_under(color=(1, 1, 1, 0))

    for zoom in ('arc', 'cty', 'ngb'):
        osgb = ccrs.epsg(27700)
        fig, ax = plt.subplots(subplot_kw={'projection':osgb}, figsize=(12, 13))
        ax.set_frame_on(False)
        ax = plt.axes([0, 0.07, 1, 1], projection=osgb)
        ax.set_extent(EXTENTS[zoom], crs=osgb)

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


def dev_plot(fname, dev_fname, norm, arc_mask):
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

    # TODO check filename pattern
    year, rate, dwellings, policy, _ = os.path.basename(fname).split('_')

    if dwellings == 'expansion':
        dwellings = 'exp'
    elif dwellings == 'settlements':
        dwellings = 'set'
    else:
        assert False, dwellings

    out_name = f"dwellings_{policy}_{dwellings}_{rate}_{year}_zoom.png"

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

        plt.savefig(out_name.replace('zoom', zoom))

        fig = plt.gcf()
        fig.clf()
        plt.close()


if __name__ == '__main__':
    with open("config.json", 'r') as fh:
        base_folder = json.load(fh)['data_folder']
    main(base_folder)
