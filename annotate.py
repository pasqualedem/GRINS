import torch
import pandas as pd
import numpy as np
import click
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from matplotlib import cm
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)

# ============================================================
# MODEL LOADERS
# ============================================================
def load_model(mode):
    if mode == "instance":
        checkpoint = "facebook/mask2former-swin-large-coco-instance"
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
    else:
        checkpoint = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device

# ============================================================
# IMAGE UTILITIES
# ============================================================
def get_overlay(original_img, segmentation, alpha=0.4):
    unique_ids = np.unique(segmentation)
    cmap = cm.get_cmap("tab20", len(unique_ids))
    
    id_to_color = {
        int(k): (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        for i, k in enumerate(unique_ids)
    }

    h, w = segmentation.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in id_to_color.items():
        mask_rgb[segmentation == k] = color
    
    mask_img = Image.fromarray(mask_rgb)
    return Image.blend(original_img, mask_img, alpha)

# ============================================================
# PROCESSING LOGIC
# ============================================================
def process_subfolder(input_folder, output_base, processor, model, device, mode, alpha):
    """Processes images and returns a list of dictionaries containing stats."""
    image_files = [
        p for p in input_folder.glob("*") 
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    
    if not image_files:
        return []

    folder_name = input_folder.name
    annotated_dir = output_base / folder_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    
    id2label = model.config.id2label
    folder_results = []

    click.echo(f"\n📂 Processing folder: {folder_name}")
    for img_path in tqdm(image_files, desc=f"  Annotating {folder_name}", leave=False):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = model(**inputs)

        if mode == "instance":
            result = processor.post_process_instance_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
            segmentation = result["segmentation"].cpu().numpy()
            h, w = segmentation.shape
            total_pixels = h * w
            counts = {}
            for seg in result["segments_info"]:
                label = id2label[seg["label_id"]]
                mask_sum = (segmentation == seg["id"]).sum()
                counts[label] = counts.get(label, 0) + int(mask_sum)
            
            stats = {"subfolder": folder_name, "filename": img_path.name}
            for label, pixels in counts.items():
                stats[label] = (pixels / total_pixels) * 100
        else:
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False
            )
            segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            h, w = segmentation.shape
            total_pixels = h * w
            stats = {"subfolder": folder_name, "filename": img_path.name}
            unique_classes, counts = np.unique(segmentation, return_counts=True)
            for class_id, count in zip(unique_classes, counts):
                label = id2label[class_id]
                stats[label] = (count / total_pixels) * 100

        folder_results.append(stats)
        overlay = get_overlay(image, segmentation, alpha=alpha)
        overlay.save(annotated_dir / f"annotated_{img_path.name}")

    return folder_results

# ============================================================
# CLI ENTRY POINT
# ============================================================
@click.command()
@click.option('--root', '-r', type=click.Path(exists=True), required=True, help="Input folder with 0, 90, 180, 270.")
@click.option('--output', '-o', type=click.Path(), required=True, help="Separate output folder.")
@click.option('--mode', '-m', type=click.Choice(['semantic', 'instance']), default='semantic')
@click.option('--alpha', '-a', default=0.4)
def run_batch(root, output, mode, alpha):
    """Processes subfolders and generates a single global stats.csv."""
    root_path = Path(root)
    output_path = Path(output)
    
    subfolders = ["0", "90", "180", "270"]
    valid_dirs = [root_path / d for d in subfolders if (root_path / d).is_dir()]
    
    if not valid_dirs:
        click.echo("Error: Subfolders 0, 90, 180, 270 not found.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    processor, model, device = load_model(mode)

    all_data_accumulated = []

    for folder in valid_dirs:
        folder_stats = process_subfolder(folder, output_path, processor, model, device, mode, alpha)
        all_data_accumulated.extend(folder_stats)
    
    # Generate the Global CSV
    if all_data_accumulated:
        df = pd.DataFrame(all_data_accumulated).fillna(0)
        # Organize columns: Subfolder and Filename first, then labels
        cols = ["subfolder", "filename"] + [c for c in df.columns if c not in ["subfolder", "filename"]]
        df[cols].to_csv(output_path / "stats_global.csv", index=False)
        click.echo(f"\n✅ Global statistics saved to: {output_path / 'stats_global.csv'}")
    
    click.echo(f"🎨 Annotated images saved in subdirectories under: {output_path}")

if __name__ == "__main__":
    run_batch()