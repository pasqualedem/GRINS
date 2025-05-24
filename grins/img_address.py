
import requests
import numpy as np
from pathlib import Path
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from grins.config import API_KEY,CITY1,CITY2
from tqdm import tqdm

def is_valid_image(content):
    return b"Sorry, we have no imagery here" not in content

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
    if response.status_code == 200 and is_valid_image(response.content):
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image for location ({lat}, {lon}) with heading. Error: {response.status_code}")

def get_points():
    
    url= "https://maps.googleapis.com/maps/api/geocode/json?"
    params = {
        'address': f'{CITY1}',
        'key': API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        city_name=CITY1
        filename = Path(f"data\\city1\\coords\\geocode_{city_name}.json") 
        data = response.json()
        # with open(filename,"w") as f:
        #     json.dump(data,f,indent=2)

        viewport =data["results"][0]["geometry"]["viewport"]
        south = viewport['southwest']['lat']
        west = viewport['southwest']['lng']
        north = viewport['northeast']['lat']
        east = viewport['northeast']['lng']

        punti=genera_punti_griglia(south,north,west,east,1000)
        return punti

        # coords = f"points_of_{city_name}.json"
        # with open(coords, "w") as c:
        #     json.dump(punti,c,indent=2)
        #     print("punti salvati")
    else:
        print(f"Failed to download image for location ({CITY1}). Error: {response.status_code}")

def genera_punti_griglia(south, north, west, east, num_punti):
    righe = colonne = int(num_punti ** 0.5)
    latitudini = np.linspace(south, north, righe)
    longitudini = np.linspace(west, east, colonne)
    punti = [(lat, lon) for lat in latitudini for lon in longitudini]
    return punti

def download_images_for_heading(lat, lon, heading, image_path, filename_prefix):
    heading_dir = image_path / str(heading)

    heading_dir.mkdir(parents=True, exist_ok=True)

    filename = heading_dir / f'{filename_prefix}_heading_{heading}.jpg'

    get_street_view_image(lat, lon, heading, 0, 90, filename)

def download_images(lat, lon, image_path, filename_prefix):
    headings = [0, 90, 180, 270]

    with ThreadPoolExecutor() as executor:
        executor.map(lambda heading: download_images_for_heading(lat, lon, heading, image_path, filename_prefix), headings)


coords=get_points()

base_path = Path("data/city1")
for i,(lat,lon) in tqdm(enumerate(coords[93:])):
    filename_prefix = f'image_{i}'
    download_images(lat,lon,base_path,filename_prefix)