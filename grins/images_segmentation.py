from pathlib import Path
import pathlib
import typer
from loguru import logger
import os
import re
import ast
import torch
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm
from grins.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR

app = typer.Typer()

def merge_output_with_coordinates(output_path, coordinates_path, output_final_path):
    output_df = pd.read_csv(output_path)
    coordinates_df = pd.read_csv(coordinates_path)
    
    output_df['image_id'] = output_df['image_id'].astype(int)
    coordinates_df['ID'] = coordinates_df['ID'].astype(int)
    
    def extract_first_id(path):
        match = re.search(r'(\d+)_', str(path))
        return int(match.group(1)) if match else None
    
    output_df['first_id'] = output_df['path_0'].apply(extract_first_id)
    
    output_df.sort_values(by=['image_id', 'first_id'], inplace=True)
    output_df.reset_index(drop=True, inplace=True)
    
    coordinates_df.sort_values(by=['ID'], inplace=True)
    coordinates_df.reset_index(drop=True, inplace=True)
      
    output_df['lon'] = coordinates_df['lon']
    output_df['lat'] = coordinates_df['lat']
    
    output_df.drop('first_id', axis=1, inplace=True)
    
    cols = list(output_df.columns)
    cols.insert(cols.index('path_0'), cols.pop(cols.index('lon')))
    cols.insert(cols.index('path_0'), cols.pop(cols.index('lat')))
    output_df = output_df.reindex(columns=cols)
    
    output_df.to_csv(output_final_path, index=False)
    print(f"Merged file saved to: {output_final_path}")

def extract_image_ids(image_path):
    filename = os.path.basename(image_path)
    numbers = re.findall(r'(\d+)', filename)
    if len(numbers) >= 2:
        return numbers[0], numbers[1]  # sub_id, group_id
    else:
        return numbers[0], None

class StreetViewDataset(Dataset):
    def __init__(self, image_paths, color_mapping, macro_mapping, processor):
        self.image_paths = image_paths
        self.color_mapping = color_mapping
        self.macro_mapping = macro_mapping
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return image_path, image, inputs

def custom_collate(batch):
    image_paths, images, inputs = zip(*batch)
    input_dict = {key: torch.cat([inp[key] for inp in inputs], dim=0) for key in inputs[0]}
    return image_paths, list(images), input_dict

def get_image_paths(angle_dirs, base_path):
    image_paths = []
    for angle in angle_dirs:
        path_list = pathlib.Path(os.path.join(base_path, angle))
        file_list = sorted([str(path) for path in path_list.glob('*.jpg')])
        image_paths.extend(file_list)
    return image_paths

def process_batch(model, batch, color_mapping, macro_mapping, device, processor):
    image_paths, images, inputs = batch
    inputs = {k: inputs[k].to(device) for k in inputs}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_semantic_segmentation(
        outputs, 
        target_sizes=[img.size[::-1] for img in images]
    )
    
    blended_images = []
    pixel_distributions = []
    for img, mask in zip(images, results):
        mask = mask.cpu().numpy()
        output_image_np = np.array(img)
        
        for label in np.unique(mask):
            text_label = model.config.id2label.get(label, str(label))
            if text_label in color_mapping:
                output_image_np[mask == label] = color_mapping[text_label]
        
        blended_image = Image.blend(img, Image.fromarray(output_image_np), alpha=0.7)
        blended_images.append(blended_image)
        
        pixel_distribution = defaultdict(int)
        for label in np.unique(mask):
            text_label = model.config.id2label.get(label, f"unknown_{label}")
            pixel_count = np.sum(mask == label)
            pixel_distribution[macro_mapping.get(text_label, 'Unknown')] += pixel_count
        pixel_distributions.append(pixel_distribution)
    
    return image_paths, blended_images, pixel_distributions

def update_csv_data(image_paths, blended_images, pixel_distributions, output_folder, csv_data):
    for image_path, blended_image, pixel_distribution in zip(image_paths, blended_images, pixel_distributions):
        angle_dir = os.path.basename(os.path.dirname(image_path))
        angle_output_folder = os.path.join(output_folder, angle_dir)
        os.makedirs(angle_output_folder, exist_ok=True)
        output_path = os.path.join(angle_output_folder, os.path.basename(image_path))
        blended_image.save(output_path)
        
        # Extract both the sub_id and the group_id.
        sub_id, group_id = extract_image_ids(image_path)
        # Build a composite key to uniquely identify each image instance.
        composite_key = f"{sub_id}_{group_id}"
        
        if composite_key not in csv_data:
            csv_data[composite_key] = {
                "image_id": group_id,
                "sub_id": sub_id,
                "path_0": None,
                "path_90": None,
                "path_180": None,
                "path_360": None,
                "pixel_distribution": {}
            }
            
        csv_data[composite_key][f"path_{angle_dir}"] = output_path
        
        # Update the pixel distribution.
        for key, count in pixel_distribution.items():
            csv_data[composite_key]["pixel_distribution"][key] = csv_data[composite_key]["pixel_distribution"].get(key, 0) + count

@app.command()
def main(
    excel_file_path: Path = EXTERNAL_DATA_DIR / "macro_classes_with_colors.xlsx",
    image_path: Path = PROCESSED_DATA_DIR / "street_view_images",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    angle_dirs = ['0', '90', '180', '270']

    if image_path.exists():
        output_folder = image_path / 'masked_output'
        csv_path = image_path / 'output.csv'
    
    if excel_file_path.exists():
        df = pd.read_excel(excel_file_path)
        color_mapping = {
            label: ast.literal_eval(row['RGB'])
            for _, row in df.iterrows()
            for label in str(row['Original Labels']).split(', ')
        }
        macro_mapping = {
            label: row['New Class']
            for _, row in df.iterrows()
            for label in str(row['Original Labels']).split(', ')
        }
    
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-panoptic"
    ).to(device)
    
    image_paths = get_image_paths(angle_dirs, image_path)
    dataset = StreetViewDataset(image_paths, color_mapping, macro_mapping, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=32,
        shuffle=False, 
        num_workers=os.cpu_count(), 
        collate_fn=custom_collate
    )
    
    csv_data = {}
    
    for batch in tqdm(dataloader, desc="Processing Images", unit="batch"):
        image_paths_batch, blended_images, pixel_distributions = process_batch(
            model, batch, color_mapping, macro_mapping, device, processor
        )
        update_csv_data(image_paths_batch, blended_images, pixel_distributions, output_folder, csv_data)
    
    for data in csv_data.values():
        data["pixel_distribution"] = str(data["pixel_distribution"])
    
    df_csv = pd.DataFrame(list(csv_data.values()))
    df_csv.to_csv(csv_path, index=False)

    merge_output_with_coordinates(csv_path, EXTERNAL_DATA_DIR / "merged_coordinates.csv", image_path / "output_with_coordinates.csv")

if __name__ == "__main__":
    app()