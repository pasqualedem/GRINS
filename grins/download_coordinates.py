import osmnx as ox
import random
from shapely.geometry import LineString, Point
from pathlib import Path
from .config import EXTERNAL_DATA_DIR


def get_random_points_on_edges(location: str, n: int):
    """
    Extracts a graph for a given location and selects n random edges.
    For each edge, selects a random point along it in lon, lat coordinates.

    Args:
        location (str): The location string to extract the graph (e.g., "Manhattan, New York, USA").
        n (int): The number of random edges to select.

    Returns:
        list: A list of tuples containing (lon, lat) coordinates of random points.
    """
    # Step 1: Download the graph for the location
    G = ox.graph_from_place(location, network_type="drive")

    # Step 2: Select n random edges
    random_edges = random.sample(list(G.edges(data=True)), n)

    # Step 3: For each edge, select a random point along it
    random_points = []
    for u, v, data in random_edges:
        # Check if the edge has geometry
        if "geometry" in data:
            line: LineString = data["geometry"]
        else:
            # If no geometry, create a straight line between the nodes
            point_u = Point((G.nodes[u]["x"], G.nodes[u]["y"]))
            point_v = Point((G.nodes[v]["x"], G.nodes[v]["y"]))
            line = LineString([point_u, point_v])

        # Generate a random point along the line
        random_distance = random.uniform(0, line.length)
        random_point: Point = line.interpolate(random_distance)

        # Extract lon, lat coordinates
        lon, lat = random_point.x, random_point.y
        random_points.append((lon, lat))

    return random_points


def get_points_csv(location: str, n: int):
    """
    Generates random points and saves them to a CSV file.

    Args:
        location (str): The location string to extract the graph (e.g., "Manhattan, New York, USA").
        n (int): The number of random edges to select.
        output_dir (str): The directory where the CSV file will be saved.
    """
    # Generate random points
    random_points = get_random_points_on_edges(location, n)

    # Save points to CSV
    csv_path = EXTERNAL_DATA_DIR / f'coordinates_{location.replace(", ", "_")}.csv'
    with open(csv_path, "w") as f:
        f.write("ID,lon,lat\n")
        for i, (lon, lat) in enumerate(random_points, start=1):
            f.write(f"{i},{lon},{lat}\n")


get_points_csv("Manhattan, NYC", 100)
