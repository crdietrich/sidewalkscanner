"""Plot model outputs
Colin Dietrich 2019

To generate plot, run on command line:
$ python mapping.py

Requires matching images to be hosted on line.
"""

import math
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, ColorBar, LinearColorMapper
from bokeh.transform import transform
from bokeh.palettes import Spectral6
from bokeh.tile_providers import CARTODBPOSITRON, STAMEN_TERRAIN, STAMEN_TONER
import bokeh.models as bkm

# ===== SEATTLE INVENTORY =====
df_obs = pd.read_csv('city_of_seattle_2018_observations.csv')

# ===== DataCyle =====
# df_crd = pd.read_csv('image_gps.csv')  # complete full resolution
df_crd = pd.read_csv('image_gps_second_A.csv')  # decimated to second resolution
df_crd_source_dir = "http://juniperengineering.com/stream_images_2019_02_05_224x224/"
df_crd['image_filepath'] = df_crd_source_dir + df_crd.filename

# ===== Convert lat/lon to web_mercator
# derived from the Java version explained here:
# http://wiki.openstreetmap.org/wiki/Mercator
RADIUS = 6378137.0  # in meters on the equator


def lat2y(a):
    return math.log(math.tan(math.pi / 4 + math.radians(a) / 2)) * RADIUS


def lon2x(a):
    return math.radians(a) * RADIUS


df_obs['webm_longitude'] = df_obs.X.apply(lon2x)
df_obs['webm_latitude'] = df_obs.Y.apply(lat2y)

df_obs = df_obs[df_obs.SURFACE_CONDITION.notnull()]
df_obs = df_obs[df_obs.SURFACE_CONDITION.str.contains('CRACK').values]

df_crd['webm_longitude'] = df_crd.lon.apply(lon2x)
df_crd['webm_latitude'] = df_crd.lat.apply(lat2y)

tooltip_html = """
<div style="border: 2px solid black;">
<div>

    <div style="font-size: 24px;">Location: @lat @lon</div>
</div>
    <div>
        <img src="@imgs" height="400" alt="@imgs" width="400" border="1"></img>
    </div>
</div>
"""
# <div style="font-size: 15px;">Predicted Class: @pred_class</div>
# <div style="font-size: 24px;">Crack Identified</div>
# <img src="@imgs" height="224" alt="@imgs" width="224" border="1"></img>
# style="float: left; margin: 0px 15px 15px 0px;"

source_obs = ColumnDataSource(data=dict(longitude=df_obs.webm_longitude,
                                        latitude=df_obs.webm_latitude,
                                        )
                              )

source_crd = ColumnDataSource(data=dict(longitude=df_crd.webm_longitude,
                                        latitude=df_crd.webm_latitude,
                                        pred_class=df_crd.predicted,
                                        imgs=df_crd.image_filepath,
                                        lon=df_crd.lon,
                                        lat=df_crd.lat
                                        )
                              )

color_mapper_crd = LinearColorMapper(palette=Spectral6,
                                     low=0, #df_crd.predicted.min(),  # useful for multi-class
                                     high=1, #df_crd.predicted.max()  # useful for multi-class
                                     )

color_bar = ColorBar(color_mapper=color_mapper_crd, width=8,  location=(0, 0))

xmin = df_crd.webm_longitude.min()
xmin = xmin - xmin * 0.1
xmax = df_crd.webm_longitude.max()
xmax = xmax + xmax * 0.1

ymin = df_crd.webm_latitude.min()
#ymin = ymin - ymin * 0.1
ymax = df_crd.webm_latitude.max()
#ymax = ymax + ymax * 0.1

tools_to_use = "pan,zoom_in,zoom_out,wheel_zoom,box_zoom,lasso_select,undo,redo,reset,save"
# when using meractor, decimal lat/lon must be multiplied by 100,000
map_figure = figure(
           sizing_mode='stretch_both',
           x_range=(xmin, xmax),
           y_range=(ymin, ymax),
           x_axis_type="mercator", y_axis_type="mercator",
           tools=tools_to_use
           )

map_figure.title.text = "Sidewalk Scanner - A Crack Identification Tool"
map_figure.title.align = "center"
map_figure.title.text_font = "Trebuchet MS"
map_figure.title.text_color = "black"
map_figure.title.text_font_size = "24px"
map_figure.xaxis.axis_label = "Longitude"
map_figure.yaxis.axis_label = "Latitude"

# first glyphs, Seattle city data
g1 = bkm.CircleX(x='longitude', y='latitude',
                 size=20, line_color='black', fill_alpha=0.5)
g1_render = map_figure.add_glyph(source_or_glyph=source_obs, glyph=g1)

# 2nd glyph, CRD data collected
g2 = bkm.Circle(x='longitude', y='latitude',
                size=20,
                fill_alpha=0,
                line_color=transform('pred_class', color_mapper_crd))
g2_render = map_figure.add_glyph(source_or_glyph=source_crd, glyph=g2)
g2_hover = bkm.HoverTool(renderers=[g2_render],
                         tooltips=tooltip_html)
map_figure.add_tools(g2_hover)

map_figure.add_tile(STAMEN_TONER)

#map_figure.add_layout(color_bar, 'right')  # useful for multi-class

"""
color_bar_label = Label(x=-40, y=cfg.PROFILE_DETAIL_PLOT_HEIGHT / 2 - 50,
                        x_units='screen', y_units='screen',
                        text='Test Labeling',
                        text_align='center',
                        render_mode='canvas',
                        angle=270,
                        angle_units='deg',
                        )
"""

output_file("sidewalkscanner.html", title="Sidewalk Scanner v1.0")
show(map_figure)
