import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import igraph as ig
import numpy as np

from itertools import product

colors_dict = {
    "lime": "Ants",
    "red": "Queen",
    "cyan": "Nurses",
    "yellow": "Cleaners",
    "blue": "Foragers",
    "orange": "Max Centrality",
    "grey": "Disappeared",
}


def read_graph(colony, day):
    path = f"./data/insecta-ant-colony{colony}/ant_mersch_col{colony}_day{day}.graphml"
    graph = ig.Graph.Read_GraphML(path)
    graph_df = pd.DataFrame.from_dict(
        {v.index: v.attributes() for v in graph.vs}, orient="index"
    )

    return graph, graph_df


def edge_slider_update(colony, day):
    graph, graph_df = read_graph(colony, day)

    edges = list(set(graph.es["weight"]))
    return max(edges)


def remove_edges(graph, edge_filter):
    to_remove = [
        e.index
        for e in graph.es
        if e["weight"] < edge_filter[0] or e["weight"] > edge_filter[1]
    ]
    graph.delete_edges(to_remove)
    return graph


def create_color_map(ants_df, period, max_centralities=None):
    colors = []
    for i in range(ants_df.shape[0]):
        if ants_df.iloc[i][period] == "Q":
            colors.append("red")
        else:
            colors.append("lime")

        if max_centralities is not None:
            if ants_df.iloc[i]["id"] == max_centralities:
                colors[i] = "orange"

    return colors


def create_color_map_community(ants_df, period):
    colors = []
    for i in range(ants_df.shape[0]):
        if ants_df.iloc[i][period] == "N":
            colors.append("cyan")
        elif ants_df.iloc[i][period] == "F":
            colors.append("lime")
        elif ants_df.iloc[i][period] == "C":
            colors.append("yellow")
        elif ants_df.iloc[i][period] == "Q":
            colors.append("blue")
        else:
            colors.append("grey")

    return colors


def get_nodes_edges_positions(Graph, layout="kamada_kawai", period="group_period1"):
    layout = Graph.layout(layout)

    df = pd.DataFrame(columns=["id", "x", "y", "role"])

    for i in range(len(Graph.vs)):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "id": Graph.vs[i]["id"],
                        "x": layout[i][0],
                        "y": layout[i][1],
                        "role": Graph.vs[i][period],
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )
    print(df.dtypes)
    # Get node positions
    Xn = [layout[k][0] for k in range(len(Graph.vs))]
    Yn = [layout[k][1] for k in range(len(Graph.vs))]

    # Get edge positions
    Xe = []
    Ye = []
    for e in Graph.es:
        Xe.extend([layout[e.source][0], layout[e.target][0], None])
        Ye.extend([layout[e.source][1], layout[e.target][1], None])

    return Xn, Yn, Xe, Ye


def _get_degree_centrality(graph, graph_df, period):
    centrality = [val / graph.maxdegree() for val in graph.degree(graph.vs, mode="all")]
    colors = create_color_map(
        graph_df, "group_period1", graph.vs()[np.argmax(centrality)]["id"]
    )
    sizes = [val * 50 for val in centrality]

    ris = (
        "Degree Centrality: "
        + graph.vs()[np.argmax(centrality)]["id"]
        + " with value "
        + str(graph.maxdegree())
    )
    return colors, sizes, ris


def _get_betweenness_centrality(graph, graph_df, period):
    tmp = graph.betweenness()
    centrality = [val / max(tmp) for val in tmp]
    colors = create_color_map(
        graph_df, "group_period1", graph.vs()[np.argmax(centrality)]["id"]
    )
    sizes = [val * 50 for val in centrality]
    ris = (
        "Betweenness Centrality: "
        + graph.vs()[np.argmax(centrality)]["id"]
        + " with value "
        + str(max(tmp))
    )

    return colors, sizes, ris


def _get_eigenvector_centrality(graph, graph_df, period):
    centrality = graph.eigenvector_centrality(directed=False)
    colors = create_color_map(
        graph_df, "group_period1", graph.vs()[np.argmax(centrality)]["id"]
    )
    sizes = [val * 50 for val in centrality]
    ris = (
        "Eigenvector Centrality: "
        + graph.vs()[np.argmax(centrality)]["id"]
        + " with value "
        + str(max(centrality))
    )

    return colors, sizes, ris


def get_graph(colony, day, graph_type, edge_filter):
    graph, graph_df = read_graph(colony, day)
    ris = ""
    if graph_type == "community":
        colors = create_color_map_community(graph_df, "group_period1")
        sizes = [25 for i in range(len(graph.vs))]
    elif graph_type == "degree":
        colors, sizes, ris = _get_degree_centrality(graph, graph_df, "group_period1")
    elif graph_type == "betweenness":
        colors, sizes, ris = _get_betweenness_centrality(
            graph, graph_df, "group_period1"
        )
    elif graph_type == "eigenvector":
        colors, sizes, ris = _get_eigenvector_centrality(
            graph, graph_df, "group_period1"
        )
    else:
        sizes = [25 for i in range(len(graph.vs))]
        colors = ["lime" for i in range(len(graph.vs))]

    graph = remove_edges(graph, edge_filter)
    Xn, Yn, Xe, Ye = get_nodes_edges_positions(graph)
    # df, Xe, Ye = get_nodes_edges_positions(graph)

    edge_trace = go.Scatter(
        x=Xe,
        y=Ye,
        line=dict(width=1, color="#888"),
        hoverinfo="text",
        mode="lines",
        name="Interactions",
    )
    # Create the node trace
    node_trace = go.Scatter(
        x=Xn,
        y=Yn,
        mode="markers+text",
        textfont=dict(size=15, color="black"),
        hoverinfo="text",
        marker=dict(
            size=sizes,
            line_width=2,
            color=colors,
        ),
        name="Ants",
    )

    # Add node labels
    node_text = [f"{k['id']}" for k in graph.vs]
    node_trace.text = node_text

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Network Graph of Ants colony {colony} on day {day}",
            titlefont_size=20,
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=1000,
        ),
    )

    unique_colors = set(colors)

    for color in unique_colors:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color),
                legendgroup=color,
                showlegend=True,
                name=colors_dict[color],
            )
        )
    return fig, ris


def compute_link_sankey_diagram(df_days: pd.DataFrame):
    periods = ["Period_1", "Period_2", "Period_3", "Period_4"]
    classes = ["Nurses", "Foragers", "Cleaners", "Disappeared"]

    labels = [p + "_" + r for r, p in list(product(periods, classes))]
    colors = []
    for i in range(len(labels)):
        if classes[0] in labels[i]:
            colors.append("blue")
        elif classes[1] in labels[i]:
            colors.append("red")
        elif classes[2] in labels[i]:
            colors.append("green")
        else:
            colors.append("grey")

    dataset_periods = [
        ("group_period1", "Period_1"),
        ("group_period2", "Period_2"),
        ("group_period3", "Period_3"),
        ("group_period4", "Period_4"),
    ]
    classes_short = [
        ("Nurses", "N"),
        ("Foragers", "F"),
        ("Cleaners", "C"),
        ("Disappeared", " "),
    ]

    positions = {key: value for value, key in enumerate(labels)}
    sources = []
    targets = []
    values = []

    for i in range(1, len(dataset_periods)):
        p = df_days[[dataset_periods[i - 1][0], dataset_periods[i][0]]]
        for s_name, s_label in classes_short:
            for d_name, d_label in classes_short:
                q = f"{dataset_periods[i - 1][0]} == '{s_label}' and {dataset_periods[i][0]} == '{d_label}'"
                tmp = p.query(q)
                sources.append(positions[s_name + "_" + dataset_periods[i - 1][1]])
                targets.append(positions[d_name + "_" + dataset_periods[i][1]])
                values.append(tmp.shape[0])

    return labels, colors, sources, targets, values


def compute_node_positions(sources, targets, values):
    df = pd.DataFrame({"source": sources, "target": targets, "value": values})

    df_sum = df.groupby(["source"]).sum().reset_index()
    gruppi = df_sum.groupby(np.arange(len(df_sum)) // 4)
    node_y = []

    for _, gruppo in gruppi:
        tot = sum(gruppo["value"])
        gruppo["pos_tmp"] = gruppo["value"] / tot
        gruppo["pos"] = 1 - (gruppo["pos_tmp"].cumsum() - gruppo["pos_tmp"] / 2)
        node_y.extend(gruppo["pos"].tolist())

    tmp = df.groupby(["target"]).sum().reset_index().tail(4)
    tot = sum(tmp["value"])
    tmp["pos_tmp"] = tmp["value"] / tot
    tmp["pos"] = 1 - (tmp["pos_tmp"].cumsum() - tmp["pos_tmp"] / 2)

    node_y.extend(tmp["pos"].tolist())

    node_y = list(map(lambda x: round(x, 2), node_y))
    return node_y


def get_sankey(colony, day):
    graph, graph_df = read_graph(colony, day)
    del graph
    graph_df = graph_df[
        ["group_period1", "group_period2", "group_period3", "group_period4"]
    ]

    labels, colors, sources, targets, values = compute_link_sankey_diagram(graph_df)

    node_x = [0] * 4 + [0.33] * 4 + [0.66] * 4 + [1] * 4

    # Caloclo la posizione dei nodi
    # node_y = compute_node_positions(sources, targets, values)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=colors,
                    align="left",
                    x=node_x,
                    # y=node_y,
                ),
                link=dict(
                    arrowlen=15,
                    source=sources,
                    target=targets,
                    value=values,
                    hovertemplate="Link from node %{source.label} <br />"
                    + "to node%{target.label} <br />has value %{value}",
                ),
            )
        ]
    )

    annotations = []
    num_nodes = len(labels)
    num_groups = num_nodes // 4

    x_positions = [i / (num_groups - 1) for i in range(num_groups)]

    for i, x_position in enumerate(x_positions):
        y_position = 1.1
        label = f"Period {(i + 1)}"
        annotations.append(
            dict(
                x=x_position,
                y=y_position,
                xref="paper",
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=12, color="black"),
            )
        )

    # Aggiungi annotazioni al layout
    fig.update_layout(
        title_text="Sankey Diagram con Etichette per Gruppi di Nodi",
        annotations=annotations,
    )
    return fig