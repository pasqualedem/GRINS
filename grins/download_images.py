from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import csv
import requests
import numpy as np
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from grins.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, API_KEY,CITY1,CITY2

app = typer.Typer()


def get_street_view_image(lat, lon, heading, pitch, fov, filename):
    """Downloads a street view image from Google Maps API.

    The image is saved to the specified filename.
    The image is centered at the specified latitude and longitude,
    with the specified heading, pitch, and field of view (fov).

    Args:
        lat (_type_): latitude of the location
        lon (_type_): longitude of the location
        heading (_type_): heading of the camera in degrees
        0 is north, 90 is east, 180 is south, and 270 is west
        pitch (_type_): pitch of the camera in degrees
        fov (_type_): field of view of the camera in degrees
        filename (_type_): path to save the image
    """
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "512x512",
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": API_KEY,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        logger.info(
            f"Failed to download image for location ({lat}, {lon}) with heading {heading}. Error: {response.status_code}"
        )


def download_images_for_heading(lat, lon, heading, image_path, filename_prefix):
    """Downloads images for a specific heading (0, 90, 180, 270) for a given location.

    Args:
        lat (_type_): _description_
        lon (_type_): _description_
        heading (_type_): _description_
        image_path (_type_): _description_
        filename_prefix (_type_): _description_
    """
    heading_dir = image_path / str(heading)

    heading_dir.mkdir(parents=True, exist_ok=True)

    filename = heading_dir / f"{filename_prefix}_heading_{heading}.jpg"

    get_street_view_image(lat, lon, heading, 0, 90, filename)


def download_images(lat, lon, image_path, filename_prefix):
    """Downloads images for all headings (0, 90, 180, 270) for a given location.

    Args:
        lat (_type_): _description_
        lon (_type_): _description_
        image_path (_type_): _description_
        filename_prefix (_type_): _description_
    """
    headings = [0, 90, 180, 270]

    with ThreadPoolExecutor() as executor:
        executor.map(
            lambda heading: download_images_for_heading(
                lat, lon, heading, image_path, filename_prefix
            ),
            headings,
        )


def process_csv_and_download_images(csv_file_path, image_path):
    logger.info(f"Processing CSV file: {csv_file_path}...")
    with open(csv_file_path, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in tqdm(enumerate(reader), desc="Processing rows"):
            lat = row['lat']
            lon = row['lon']
            unique_id = row['ID']
            filename_prefix = f'image_{i}_{unique_id}'

            download_images(lat, lon, image_path, filename_prefix)


def most_frequent_pixel_intensity(image_path):
    if image_path.exists():
        img = np.array(Image.open(image_path).convert("L"))

        flattened_array = img.flatten()

        pixel_counts = np.bincount(flattened_array, minlength=256)

        most_frequent_intensity = np.argmax(pixel_counts)
        max_count = pixel_counts[most_frequent_intensity]
    else:
        logger.error(f"Image not found: {image_path}")

    return most_frequent_intensity, max_count


def check_and_delete_images(directory_path, target_intensity=227, target_count=257929):
    for filename in os.listdir(directory_path):
        file_path = directory_path / filename

        if str(file_path).lower().endswith((".png", ".jpg")):
            most_frequent_intensity, max_count = most_frequent_pixel_intensity(file_path)
            print(most_frequent_intensity, max_count)

            if most_frequent_intensity == target_intensity and max_count == target_count:
                os.remove(file_path)
                logger.info(
                    f"Deleted {file_path} (Intensity: {most_frequent_intensity}, Count: {max_count})"
                )


@app.command()

def main(csv_file_path: Path = EXTERNAL_DATA_DIR / "merged_coordinates.csv",
image_path: Path = PROCESSED_DATA_DIR / "street_view_images",
):
    image_path.mkdir(parents=True, exist_ok=True)

    if not csv_file_path.exists():
        logger.error(f"CSV file not found: {csv_file_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    process_csv_and_download_images(csv_file_path, image_path)

    subfolders = [f for f in image_path.iterdir() if f.is_dir()]
    with ThreadPoolExecutor() as executor:
        executor.map(check_and_delete_images, subfolders)


if __name__ == "__main__":
    app()
