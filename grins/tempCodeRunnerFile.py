
import requests
import numpy as np
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from grins.config import API_KEY,CITY1,CITY2


def get_street_view_image(lat, lon, heading, pitch, fov, filename):
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        'size': '512x512',
        'location': f'{lat},{lon}',
        'heading': heading,
        'pitch': pitch,
        'fov': fov,
        'key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image for location ({lat}, {lon}) with heading {heading}. Error: {response.status_code}")

def get_points():
    
    url= "https://maps.googleapis.com/maps/api/geocode/json?"
    params = {
        'address': f'{CITY1}',
        'key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        city_name=CITY1
        filename = f"geocode_{city_name}.json"
        data = response.json()
        with open(filename,"w") as f:
            json.dump(data,f,indent=2)
        viewport =data["results"][0]["geometry"]["viewport"]
        south=viewport["southwest"]["lng"]
        west=viewport["southwest"]["lat"]
        north=viewport["northeast"]["lng"]
        east=viewport["northeast"]["lat"]
        punti=genera_punti_griglia(south,north,west,east,1000)
        coords = f"points_of_{city_name}.json"
        with open(coords, "w") as c:
            json.dump(punti,c,indent=2)
            print("punti salvati")
    else:
        print(f"Failed to download image for location ({CITY1}). Error: {response.status_code}")

def genera_punti_griglia(south, north, west, east, num_punti):
    righe = colonne = int(num_punti ** 0.5)
    latitudini = np.linspace(south, north, righe)
    longitudini = np.linspace(west, east, colonne)
    punti = [(lat, lon) for lat in latitudini for lon in longitudini]
    return punti

def download_images_for_heading(lat, lon, heading, image_path):
    heading_dir = image_path / str(heading)
    print(f"qui:{heading_dir}")
    heading_dir.mkdir(parents=True, exist_ok=True)

    filename = f'city_img_{heading}.jpg'

    get_street_view_image(lat, lon, heading, 0, 90, filename)

def download_images(lat, lon, image_path):
    headings = [0, 90, 180, 270]

    with ThreadPoolExecutor() as executor:
        executor.map(lambda heading: download_images_for_heading(lat, lon, heading, image_path), headings)



import json
filename="points_of_Matera.json"
with open(filename, "r") as f:
    coords = json.load(f)
print("json loaded.")
for (lat,lon) in coords:
    
    download_images(lat,lon,"data\\city1")