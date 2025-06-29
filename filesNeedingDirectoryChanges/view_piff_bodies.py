#!/usr/bin/env python3
"""
view_piff_bodies.py

Usage:
    python view_piff_bodies.py path/to/your_file.piff

This will open a browser window with an interactive 3D view
of all voxels labeled as "body", colored by their CellID.
"""

import os
import sys
import pandas as pd
import plotly.graph_objs as go

def main(piff_path):
    # 1) Load the PIFF file

    if os.path.exists(piff_path):
        print("opening ", piff_path)
    else:
        return
    # Each line: CellID CellType x1 x2 y1 y2 z1 z2
    df = pd.read_csv(
        piff_path,
        sep=r"\s+",
        header=None,
        names=["CellID","CellType","x1","x2","y1","y2","z1","z2"]
    )

    # 2) Filter to body voxels only
    body_df = df #df[df.CellType == "Body"]

    # 3) Extract coordinates and IDs
    xs = body_df["x1"].values
    ys = body_df["y1"].values
    zs = body_df["z1"].values
    ids = body_df["CellID"].values

    # 4) Build a 3D scatter
    scatter = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(
            size=2,
            opacity=0.6,
            color=ids,
            colorscale="Viridis",
            colorbar=dict(title="CellID"),
        )
    )

    layout = go.Layout(
        title="3D View of Body Voxels",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
