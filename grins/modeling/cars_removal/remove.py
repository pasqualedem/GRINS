# ...existing code...

import os
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import click
import cv2

try:
	from tqdm import tqdm
except ImportError:
	tqdm = lambda x, **kwargs: x

# Import LAMA inpainting
try:
	from simple_lama_inpainting import SimpleLama
except ImportError:
	raise ImportError("Please install simple_lama_inpainting (pip install simple-lama-inpainting)")

def mask_path_to_pil(mask_path):
	mask = Image.open(mask_path).convert("L")
	return mask

def perimeter_based_dilation(mask_pil):
	mask = np.array(mask_pil)
	mask_binary = (mask > 127).astype(np.uint8)
	contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	total_perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
	if total_perimeter < 200:
		kernel_size = 5
		iterations = 3
	elif total_perimeter < 400:
		kernel_size = 7
		iterations = 3
	elif total_perimeter < 700:
		kernel_size = 9
		iterations = 3
	elif total_perimeter < 1200:
		kernel_size = 11
		iterations = 3
	else:
		kernel_size = 13
		iterations = 3
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	dilated = cv2.dilate(mask_binary * 255, kernel, iterations=iterations)
	return Image.fromarray(dilated, mode="L")

@click.command()
@click.argument("root_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output-dir", default=None, help="Output directory for car-removed images")
@click.option("--device", default="cuda:0", help="Torch device (not used for LAMA, but for future use)")
@click.option("--resume", is_flag=True, help="Resume from existing car_removal_annotations.json if present")
def remove_cars(root_dir, output_dir, device, resume):
	"""
	Remove cars from images using LAMA inpainting, based on car_annotations.json.
	"""
	root = Path(root_dir)
	images_dir = root / "images"
	car_removal_dir = root / "car_removal" if output_dir is None else Path(output_dir)
	car_removal_dir.mkdir(parents=True, exist_ok=True)
	masks_dir = car_removal_dir / "masks"
	annotation_path = car_removal_dir / "car_annotations.json"
	if not annotation_path.exists():
		raise FileNotFoundError(f"car_annotations.json not found in {car_removal_dir}")

	removal_ann_path = car_removal_dir / "car_removal_annotations.json"

	# Load annotations
	with open(annotation_path, "r") as f:
		annotations = json.load(f)

	# Resume logic
	if resume and removal_ann_path.exists():
		with open(removal_ann_path, "r") as f:
			removal_annotations = json.load(f)
	else:
		removal_annotations = {}

	simple_lama = SimpleLama()

	for rel_img_path, car_list in tqdm(annotations.items(), desc="Images"):
		# Skip images with no car annotations
		if not car_list:
			continue
		img_out_dir = car_removal_dir / Path(rel_img_path).parent
		img_out_dir.mkdir(parents=True, exist_ok=True)
		img_path = images_dir / rel_img_path
		if not img_path.exists():
			print(f"Image not found: {img_path}")
			continue

		# Load original image
		current_image = Image.open(img_path).convert("RGB")
		removal_steps = []
		# Remove cars in order
		for idx, car_ann in enumerate(car_list):
			mask_path = car_ann.get("base_mask_path")
			if not mask_path or not os.path.exists(mask_path):
				print(f"Mask not found: {mask_path}")
				continue
			mask_pil = mask_path_to_pil(mask_path)
			dilated_mask_pil = perimeter_based_dilation(mask_pil)
			edited_image = simple_lama(current_image, dilated_mask_pil)
			# Save intermediate result
			removal_id = f"{Path(rel_img_path).stem}_removal_{idx}"
			out_img_path = img_out_dir / f"{removal_id}.png"
			edited_image.save(out_img_path)
			# Record annotation
			removal_steps.append({
				"removal_id": removal_id,
				"car_annotation_index": idx,
				"car_annotation": car_ann,
				"output_image_path": str(out_img_path.relative_to(car_removal_dir)),
			})
			# Update current image for next removal
			current_image = edited_image
		# Only save annotation if there were removals
		if removal_steps:
			removal_annotations[rel_img_path] = {
				"removal_steps": removal_steps
			}
			# Save after each image for safety
			with open(removal_ann_path, "w") as f:
				json.dump(removal_annotations, f, indent=2)

if __name__ == "__main__":
	remove_cars()
