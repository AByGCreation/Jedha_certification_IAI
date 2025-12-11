
import platform
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import colors
import numpy as np

#---- Jedha Colors for plots ----

jedhaColor_violet = '#8409FF'
jedhaColor_blue = '#3AE5FF'
jedhaColor_blueLight = '#89C2FF'
jedhaColor_white = '#DFF4F5'
jedhaColor_black = '#170035'

jedha_bg_color = jedhaColor_white
jedha_grid_color = jedhaColor_black

if platform.system() == "Darwin":
    jedha_font = "Avenir Next"
else:
    jedha_font = "Avenir Next, Arial, sans-serif, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol"

# Plotly Jedha Template
pio.templates["jedha_template"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=jedha_font, color=jedhaColor_black),
        title=dict(x=0.5, xanchor="center", font=dict(size=24, color=jedhaColor_black)),
        plot_bgcolor=jedha_bg_color,
        paper_bgcolor=jedha_bg_color,
        xaxis=dict(
            gridcolor=jedha_grid_color,
            zerolinecolor=jedha_grid_color,
            linecolor=jedha_grid_color,
            ticks="outside",
            tickcolor=jedha_grid_color,
        ),
        yaxis=dict(
            gridcolor=jedha_grid_color,
            zerolinecolor=jedha_grid_color,
            linecolor=jedha_grid_color,
            ticks="outside",
            tickcolor=jedha_grid_color,
        ),
        legend=dict(
            bgcolor=jedha_bg_color,
            bordercolor=jedha_grid_color,
            borderwidth=1,
        ),
    )
)
pio.templates.default = "jedha_template"

jedha_colors = np.array([(132, 9, 255), (223,244,245), (58, 229, 255)])/255.
jedhaCM = colors.LinearSegmentedColormap.from_list('Jedha Scale', jedha_colors)
jedhaCMInverted = colors.LinearSegmentedColormap.from_list('Jedha Scale', jedha_colors[::-1])