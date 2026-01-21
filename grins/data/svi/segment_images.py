import os
import re
import json
import click
import torch
import pathlib
import ast
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_REPO = "facebook/mask2former-swin-small-coco-panoptic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------


def parse_filename(filename):
    """
    Parses filenames in the format: image_{seq_id}_{point_id}_heading_{angle}.png
    Example: image_0_20953_heading_0.png
    """
    # Regex to capture: seq_id, point_id, angle
    match = re.search(r"image_(\d+)_(\d+)_heading_(\d+)", filename)
    if match:
        return {
            "seq_id": match.group(1),
            "point_id": match.group(2),
            "heading": match.group(3),
        }
    return None


def load_class_mappings(excel_path):
    """
    Loads color and macro-class mappings from an Excel file.
    Expected Excel columns: 'Original Labels', 'RGB', 'New Class'
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Mapping file not found: {excel_path}")

    df = pd.read_excel(excel_path)

    color_map = {}
    macro_map = {}

    # Iterate rows and split 'Original Labels' (e.g., "car, truck" -> ["car", "truck"])
    for _, row in df.iterrows():
        labels = str(row["Original Labels"]).split(", ")
        rgb_val = ast.literal_eval(row["RGB"])
        macro_class = row["New Class"]

        for label in labels:
            color_map[label] = rgb_val
            macro_map[label] = macro_class

    return color_map, macro_map


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------


class StreetViewDataset(Dataset):
    def __init__(self, image_files, processor):
        """
        Args:
            image_files (list): List of full file paths to images.
            processor: HuggingFace AutoImageProcessor.
        """
        self.image_files = image_files
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        image = Image.open(path).convert("RGB")
        # Preprocess image for the model
        inputs = self.processor(images=image, return_tensors="pt")
        # Return path as string to track metadata
        return str(path), image, inputs


def custom_collate(batch):
    """
    Custom collator to handle the dictionary output of the processor
    and list of strings (paths).
    """
    paths, images, inputs_list = zip(*batch)
    # Stack the tensor inputs (pixel_values, etc.)
    input_dict = {
        k: torch.cat([inp[k] for inp in inputs_list], dim=0) for k in inputs_list[0]
    }
    return paths, list(images), input_dict


# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------


def process_batch(model, processor, batch, device, color_mapping, macro_mapping):
    """
    Runs inference on a batch and computes pixel distributions.
    """
    paths, images, inputs = batch

    # Move inputs to GPU/Device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process (convert logits to segmentation maps)
    # target_sizes ensures masks match original image resolution
    results = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1] for img in images]
    )

    batch_results = []

    for path, img, mask in zip(paths, images, results):
        mask_np = mask.cpu().numpy()

        # --- 1. Compute Statistics ---
        raw_counts = defaultdict(int)
        macro_counts = defaultdict(int)

        unique_labels = np.unique(mask_np)

        for label_id in unique_labels:
            # Get text label from model config (e.g., 0 -> "road")
            label_text = model.config.id2label.get(label_id, str(label_id))

            # Count pixels
            count = int(np.sum(mask_np == label_id))

            # Update Raw counts
            raw_counts[label_text] += count

            # Update Macro counts
            macro_class = macro_mapping.get(label_text, "Unknown")
            macro_counts[macro_class] += count

        # --- 2. Create Blended Image (Optional Visualization) ---
        # Create a colored mask overlay
        output_image_np = np.array(img)
        for label_id in unique_labels:
            label_text = model.config.id2label.get(label_id, str(label_id))
            if label_text in color_mapping:
                # Apply color to the mask area
                output_image_np[mask_np == label_id] = color_mapping[label_text]

        # Blend original with mask (alpha 0.7)
        blended = Image.blend(img, Image.fromarray(output_image_np), alpha=0.7)

        batch_results.append(
            {
                "path": path,
                "blended_image": blended,
                "raw_distribution": dict(raw_counts),
                "macro_distribution": dict(macro_counts),
            }
        )

    return batch_results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@click.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Root folder containing heading subfolders (0, 90, 180, 270).",
)
@click.option(
    "--mapping-file",
    required=True,
    type=click.Path(exists=True),
    help="Excel file defining class colors and macro mappings.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Folder to save results (CSV, JSON, Annotated Images).",
)
@click.option(
    "--batch-size",
    default=8,
    help="Batch size for inference. Reduce if OOM error occurs.",
)
@click.option(
    "--save-images/--no-save-images",
    default=True,
    help="Whether to save the annotated overlay images.",
)
def main(input_dir, mapping_file, output_dir, batch_size, save_images):
    """
    Annotates Street View images using Mask2Former.
    Produces a CSV of pixel distributions, a JSON dump, and optionally annotated images.
    """

    # 1. Setup
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Running on device: {DEVICE}")

    # 2. Load Resources
    try:
        color_mapping, macro_mapping = load_class_mappings(mapping_file)
        click.echo("Mappings loaded successfully.")
    except Exception as e:
        click.echo(f"Error loading mappings: {e}", err=True)
        return

    processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_REPO).to(DEVICE)

    # 3. Gather Images
    # We look for images in the subdirectories 0, 90, 180, 270
    target_angles = ["0", "90", "180", "270"]
    image_paths = []

    for angle in target_angles:
        angle_dir = input_path / angle
        if angle_dir.exists():
            # Find all png/jpg files
            found = sorted(
                list(angle_dir.glob("*.png")) + list(angle_dir.glob("*.jpg"))
            )
            image_paths.extend(found)

    if not image_paths:
        click.echo("No images found in 0/90/180/270 subdirectories.", err=True)
        return

    click.echo(f"Found {len(image_paths)} images to process.")

    # 4. Process Loop
    dataset = StreetViewDataset(image_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() or 4,
        collate_fn=custom_collate,
    )

    final_records = []

    # Process in batches
    for batch in tqdm(dataloader, desc="Segmenting"):
        batch_results = process_batch(
            model, processor, batch, DEVICE, color_mapping, macro_mapping
        )

        for res in batch_results:
            original_path = pathlib.Path(res["path"])
            filename = original_path.name

            # Parse Metadata
            meta = parse_filename(filename)
            if not meta:
                # Fallback if regex fails
                seq_id, point_id, heading = (
                    "unknown",
                    "unknown",
                    original_path.parent.name,
                )
            else:
                seq_id, point_id, heading = (
                    meta["seq_id"],
                    meta["point_id"],
                    meta["heading"],
                )

            # Prepare Record
            record = {
                "seq_id": seq_id,
                "point_id": point_id,
                "heading": heading,
                "filename_relative": str(original_path.relative_to(input_path)),
                "raw_distribution": res["raw_distribution"],
                "macro_distribution": res["macro_distribution"],
            }
            final_records.append(record)

            # Save Image (Optional)
            if save_images:
                # Replicate folder structure in output
                save_dir = output_path / "annotated_images" / heading
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / filename
                res["blended_image"].save(save_path)

    # 5. Export Results

    # A. Save Full JSON (The "Big JSON" to avoid re-running)
    json_path = output_path / "segmentation_results.json"
    with open(json_path, "w") as f:
        json.dump(final_records, f, indent=4)
    click.echo(f"Saved full JSON results to {json_path}")

    # B. Save Flattened CSV/Excel
    # We need to flatten the distribution dictionaries into columns
    flat_data = []
    for r in final_records:
        row = {
            "seq_id": r["seq_id"],
            "point_id": r["point_id"],
            "heading": r["heading"],
            "filename": r["filename_relative"],
        }
        # Add Macro Columns (prefix with macro_)
        for k, v in r["macro_distribution"].items():
            row[f"macro_{k}"] = v

        # Add Raw Columns (prefix with class_)
        for k, v in r["raw_distribution"].items():
            row[f"class_{k}"] = v

        flat_data.append(row)

    df = pd.DataFrame(flat_data)

    # Fill NaNs with 0 (for classes present in some images but not others)
    df.fillna(0, inplace=True)

    # Sort for tidiness
    df.sort_values(by=["seq_id", "point_id", "heading"], inplace=True)

    csv_path = output_path / "segmentation_stats.csv"
    excel_path = output_path / "segmentation_stats.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    click.echo(f"Saved stats to {csv_path} and {excel_path}")
    click.echo("Done.")


if __name__ == "__main__":
    main()
