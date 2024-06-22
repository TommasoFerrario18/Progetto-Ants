from dash.dependencies import Input, Output
from plotly.graph_objs import *
from itertools import product
from dash import html
from dash import dcc

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.express as px
import chart_studio
import igraph as ig
import pandas as pd
import numpy as np
import dash

from DashLogic import edge_slider_update, get_graph, get_sankey


# Variabili globali
colonies = ["1", "2", "3", "4", "5", "6"]
list_periods = ["group_period1", "group_period2", "group_period3", "group_period4"]
days = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
]

colony_idx = 0
day_idx = 0

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)


# Layout
app.layout = html.Div(
    [
        html.H1("Ant Colony Graphs"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label(
                            "Colony", style={"font-weight": "bold", "margin": "10px"}
                        ),
                        dcc.Dropdown(
                            id="colony-dropdown",
                            options=[
                                {"label": "Colony" + colony, "value": colony}
                                for colony in colonies
                            ],
                            value="1",
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.Label(
                            "Days", style={"font-weight": "bold", "margin": "10px"}
                        ),
                        dcc.Slider(
                            id="day-slider",
                            min=1,
                            max=41,
                            step=1,
                            value=1,
                        ),
                    ],
                    width=6,
                ),
            ],
            style={"margin": "30px"},
        ),
        html.Div(
            id="graph-container",
            style={"margin": "50px", "border": "1px solid", "border-radius": "10px"},
            children=[
                html.H2("Graph", style={"margin": "10px"}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Select Type"),
                                dcc.RadioItems(
                                    id="graph-type",
                                    options=[
                                        {"label": "Normal", "value": "normal"},
                                        {
                                            "label": "Degree centrality",
                                            "value": "degree",
                                        },
                                        {
                                            "label": "Betweenness Centrality",
                                            "value": "betweenness",
                                        },
                                        {
                                            "label": "Eigenvector Centrality",
                                            "value": "eigenvector",
                                        },
                                        {"label": "Community", "value": "community"},
                                        {
                                            "label": "Our community",
                                            "value": "our_community",
                                        },
                                    ],
                                    value="normal",
                                    labelStyle={
                                        "display": "block",
                                    },
                                ),
                            ],
                            width=2,
                        ),
                        dbc.Col(
                            [
                                dcc.Graph(id="graph"),
                            ],
                            width=10,
                        ),
                    ],
                    style={"margin": "15px"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Edge filter"),
                                dcc.RangeSlider(
                                    id="edge-slider",
                                    min=0,
                                    max=1,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.P("Info box", id="info-box"),
                            ],
                            width=5,
                        ),
                    ],
                    style={"margin": "15px"},
                ),
            ],
        ),
        html.Hr(),
        html.Div(
            id="sankey-container",
            style={"margin": "50px", "border": "1px solid", "border-radius": "10px"},
            children=[
                html.H2("Community evolution", style={"margin": "10px"}),
                dcc.Graph(id="sankey"),
            ],
        ),
    ],
)


# Callbacks
@app.callback(
    Output("day-slider", "marks"),
    Output("day-slider", "max"),
    Output("day-slider", "value"),
    Output("day-slider", "step"),
    Input("colony-dropdown", "value"),
)
def update_day_slider(colony):
    if colony == "6":
        return {i: days[i - 1] for i in range(1, 40)}, 39, 1, 1
    return {i: days[i - 1] for i in range(1, 42)}, 41, 1, 1


@app.callback(
    Output("graph", "figure"),
    Output("info-box", "children"),
    Input("colony-dropdown", "value"),
    Input("day-slider", "value"),
    Input("graph-type", "value"),
    Input("edge-slider", "value"),
)
def update_graph(colony, day, graph_type, edge_filter):
    fig, ris = get_graph(colony, days[int(day) - 1], graph_type, edge_filter)
    return fig, ris


@app.callback(
    Output("edge-slider", "marks"),
    Output("edge-slider", "value"),
    Output("edge-slider", "step"),
    Output("edge-slider", "max"),
    Input("colony-dropdown", "value"),
    Input("day-slider", "value"),
)
def update_edge_slider(colony, day):
    max_value = edge_slider_update(colony, days[int(day) - 1])
    return None, [0, max_value], 1, max_value


@app.callback(
    Output("sankey", "figure"),
    Input("colony-dropdown", "value"),
)
def update_sankey(colony):
    return get_sankey(colony, days[0])


if __name__ == "__main__":
    app.run_server(debug=True)
