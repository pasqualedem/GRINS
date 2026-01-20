import os
import json
from pathlib import Path
import click
from PIL import Image
import torch
import numpy as np
from transformers import Sam3Processor, Sam3Model
import cv2

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def save_mask(mask_np, path):
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(path)
    return str(path)


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "root_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--device", default="cuda:0", help="Torch device")
@click.option("--batch-size", default=1, type=int, help="Number of images to process at once")
def annotate(root_dir, device, batch_size):
    """
    Annotate images in a dataset directory using SAM3.
    """
    root = Path(root_dir)
    images_dir = root / "images"
    masks_dir = root / "masks"
    annotation_path = root / "car_annotations.json"
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    obj_types = ["car", "truck", "bus", "van"]

    # Resume logic: load existing annotations if present
    if annotation_path.exists():
        with open(annotation_path, "r") as f:
            try:
                annotations = json.load(f)
            except Exception:
                annotations = {}
    else:
        annotations = {}

    for subdir in tqdm(sorted(images_dir.iterdir()), desc="Subdirs"):
        if not subdir.is_dir():
            continue
        img_list = sorted(subdir.glob("*.jpg"))
        # Filter out already processed images
        img_list = [img_path for img_path in img_list if str(img_path.relative_to(images_dir)) not in annotations]
        num_batches = (len(img_list) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(img_list), batch_size), desc=f"Batches in {subdir.name}", total=num_batches):
            batch_imgs = img_list[batch_start:batch_start + batch_size]
            if not batch_imgs:
                continue
            images_batch = []
            prompts_batch = []
            img_rels = []
            for img_path in batch_imgs:
                image = Image.open(img_path).convert("RGB")
                # Repeat each image 4 times (one per object type)
                images_batch.extend([image] * len(obj_types))
                prompts_batch.extend(obj_types)
                img_rels.extend([str(img_path.relative_to(images_dir))] * len(obj_types))
            with torch.no_grad():
                inputs = processor(
                    images=images_batch, text=prompts_batch, return_tensors="pt"
                ).to(device)
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )
            # Group results by image
            for i, img_path in enumerate(batch_imgs):
                img_rel = str(img_path.relative_to(images_dir))
                img_ann = []
                for j, obj in enumerate(obj_types):
                    idx = i * len(obj_types) + j
                    result = results[idx]
                    masks = result["masks"]
                    scores = result["scores"]
                    boxes = result["boxes"]
                    for k in range(len(masks)):
                        instance = {
                            "label": obj,
                            "box": boxes[k].cpu().tolist() if k < len(boxes) else None,
                            "base_mask_path": None,
                            "score": float(scores[k].cpu().item()) if k < len(scores) else None,
                        }
                        # Save base mask
                        base_mask_path = (
                            masks_dir
                            / Path(img_rel).parent
                            / f"{img_path.stem}_obj{j}_{k}_base.png"
                        )
                        instance["base_mask_path"] = save_mask(masks[k].cpu().numpy(), base_mask_path)
                        img_ann.append(instance)
                annotations[img_rel] = img_ann
            # Save/append after each batch
            with open(annotation_path, "w") as f:
                json.dump(annotations, f, indent=2)


if __name__ == "__main__":
    cli()
