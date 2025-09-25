import os
import random
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, Point
from pathlib import Path
from ...config import EXTERNAL_DATA_DIR  # adjust if needed

# reproducibility
random.seed(42)
np.random.seed(42)


def _get_random_points_on_edges(G, n):
    """
    Select n random edges and return random lon/lat points on them.
    Works on unprojected graph (degrees).
    """
    random_edges = random.sample(list(G.edges(data=True)), n)
    points = []
    for u, v, data in random_edges:
        if "geometry" in data and data["geometry"] is not None:
            line: LineString = data["geometry"]
        else:
            point_u = Point((G.nodes[u]["x"], G.nodes[u]["y"]))
            point_v = Point((G.nodes[v]["x"], G.nodes[v]["y"]))
            line = LineString([point_u, point_v])

        if line.length == 0:
            continue

        random_distance = random.uniform(0, line.length)
        p: Point = line.interpolate(random_distance)
        points.append((p.x, p.y))
    return points


def _get_points_with_spacing(location, spacing_m=40.0, offset_m=15.0, n_points=None):
    """
    Place points along every edge of the graph using spacing and offset.
    Works on projected graph (meters).
    """
    # Download and project
    G = ox.graph_from_place(location, network_type="drive")
    G_proj = ox.project_graph(G)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)
    edges_gdf = edges_gdf.reset_index()

    # Ensure every edge has geometry
    def ensure_geometry(row):
        if row.geometry is None or row.geometry.is_empty:
            pu = nodes_gdf.loc[row.u].geometry
            pv = nodes_gdf.loc[row.v].geometry
            return LineString([pu, pv])
        return row.geometry

    edges_gdf["geometry"] = edges_gdf.apply(ensure_geometry, axis=1)
    edges_gdf["edge_uid"] = edges_gdf.index.astype(str)

    pts, meta = [], []
    for idx, row in edges_gdf.iterrows():
        line: LineString = row.geometry
        L = line.length
        if L <= 2 * offset_m:
            continue
        distances = np.arange(offset_m, L - offset_m + 1e-8, spacing_m)
        for d in distances:
            pt = line.interpolate(d)
            pts.append(pt)
            meta.append(
                {
                    "edge_uid": row["edge_uid"],
                    "u": row["u"],
                    "v": row["v"],
                    "key": row.get("key", 0),
                    "dist_along_m": float(d),
                    "edge_length_m": float(L),
                }
            )
    if not pts:
        return []

    pts_gdf = gpd.GeoDataFrame(meta, geometry=pts, crs=edges_gdf.crs)
    pts_wgs = pts_gdf.to_crs(epsg=4326)
    points = [(i, p.x, p.y) for i, p in enumerate(pts_wgs.geometry, start=1)]
    
    if n_points is not None and len(points) > n_points:
        points = random.sample(points, n_points)
    return points


def get_points_csv(
    location: str,
    method: str = "random",
    n_points: int = None,
    spacing_m: float = 40.0,
    offset_m: float = 15.0,
):
    """
    Generate points on OSM data and save to CSV.

    Args:
        location (str): Place name (e.g., "Bari, Italy").
        method (str): "random", "spacing", "grid", or "fixed".
        n_points (int): Number of points (used for 'random' or 'fixed').
        spacing_m (float): Spacing distance (used in spacing and grid).
        offset_m (float): Offset distance (only used for spacing).
    """
    if method == "random":
        G = ox.graph_from_place(location, network_type="drive")
        assert n_points is not None, "n_points must be specified for 'random' method"
        points = _get_random_points_on_edges(G, n_points)
    elif method == "spacing":
        points = _get_points_with_spacing(location, spacing_m, offset_m, n_points)
    else:
        raise ValueError("method must be 'random' or 'spacing'")

    # Save to CSV
    csv_path = (
        EXTERNAL_DATA_DIR / f'coordinates_{location.replace(", ", "_")}_{method}.csv'
    )
    with open(csv_path, "w") as f:
        f.write("ID,lon,lat\n")
        for i, lon, lat in points:
            f.write(f"{i},{lon},{lat}\n")

    print(f"Saved {len(points)} points to {csv_path}")
    return csv_path


if __name__ == "__main__":
    # Example usage
    get_points_csv("Bari, Italy", method="spacing", n_points=1000)
