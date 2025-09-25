import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box
import osmnx as ox
from pathlib import Path
import pandas as pd
from ...config import EXTERNAL_DATA_DIR  # adjust if needed


def plot_points_from_csv(location: str, csv_path: Path, zoom_m: float = 2500.0):
    """
    Plot points saved in a CSV against the OSM street network.

    Args:
        location (str): Place name (e.g., "Bari, Italy").
        csv_path (Path): Path to CSV file with columns ID, lon, lat.
        zoom_m (float): Half-width of the zoom box in meters.
    """
    # --- Load CSV into GeoDataFrame ---
    df = pd.read_csv(csv_path)
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df.lon, df.lat)],
        crs="EPSG:4326",
    )

    # --- Load street network and project ---
    G = ox.graph_from_place(location, network_type="drive")
    G_proj = ox.project_graph(G)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
    edges_wgs = edges_gdf.to_crs(epsg=4326)

    # --- Full city plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    edges_wgs.plot(ax=axes[0], linewidth=0.5, color="lightgray")
    gdf_pts.plot(ax=axes[0], markersize=2, color="green")
    axes[0].set_title(f"{location}\nAll points vs. street network")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # --- Zoom plot (centered on city centroid in projected CRS) ---
    bounds = edges_gdf.total_bounds  # minx, miny, maxx, maxy in projected CRS
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0
    zoom_box_proj = box(cx - zoom_m, cy - zoom_m, cx + zoom_m, cy + zoom_m)
    zoom_gdf = gpd.GeoDataFrame(geometry=[zoom_box_proj], crs=edges_gdf.crs).to_crs(
        epsg=4326
    )
    minx, miny, maxx, maxy = zoom_gdf.total_bounds

    # select features within zoom area
    edges_zoom = edges_wgs[edges_wgs.intersects(zoom_gdf.geometry.iloc[0])]
    pts_zoom = gdf_pts[gdf_pts.intersects(zoom_gdf.geometry.iloc[0])]

    edges_zoom.plot(ax=axes[1], linewidth=0.6, color="lightgray")
    pts_zoom.plot(ax=axes[1], markersize=6, color="green")
    axes[1].set_xlim(minx, maxx)
    axes[1].set_ylim(miny, maxy)
    axes[1].set_title(f"{location}\nZoomed view (~{int(zoom_m)} m radius)")
    axes[1].set_xlabel("Longitude")

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    # Example usage
    location = "Bari, Italy"
    csv_path = EXTERNAL_DATA_DIR / "coordinates_Bari_Italy_spacing.csv"
    fig = plot_points_from_csv(location, csv_path, zoom_m=2500.0)
    fig_path = EXTERNAL_DATA_DIR / "plot_Bari_Italy.png"
    fig.savefig(fig_path, dpi=300)
    print(f"Saved figure to {fig_path}")
