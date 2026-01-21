import os
import numpy as np
import streamlit as st
import torch
from PIL import Image

import pandas as pd
import plotly.express as px

from matplotlib.colors import to_hex
from matplotlib import cm

from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
)

from grins.config import PROCESSED_DATA_DIR, SVI_DATA_DIR

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(layout="wide")

IMAGE_FOLDER = SVI_DATA_DIR / "Bari_Italy" / "images" / "0"

# ============================================================
# MODEL LOADERS
# ============================================================


@st.cache_resource
def load_mask2former():
    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-coco-instance"
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-coco-instance"
    )
    model.eval()
    return processor, model


@st.cache_resource
def load_segformer():
    processor = AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )
    model.eval()
    return processor, model


# ============================================================
# IMAGE LOADING
# ============================================================


def load_images(folder):
    images = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(("jpg", "jpeg", "png")):
            img = Image.open(folder / fname).convert("RGB")
            images.append({"name": fname, "image": img})
    return images


# ============================================================
# INFERENCE
# ============================================================


def run_instance(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    return processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
    )[0]


def run_semantic(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)

    return outputs.logits.argmax(dim=1)[0].cpu().numpy()


# ============================================================
# STATISTICS
# ============================================================


def instance_stats(segmentation, segments_info, id2label):
    h, w = segmentation.shape
    total = h * w

    counts = {}
    for seg in segments_info:
        mask = segmentation == seg["id"]
        label = id2label[seg["label_id"]]
        counts[label] = counts.get(label, 0) + int(mask.sum())

    return sorted(
        [(k, 100 * v / total) for k, v in counts.items()],
        key=lambda x: x[1],
        reverse=True,
    )


def semantic_stats(segmentation, id2label):
    h, w = segmentation.shape
    total = h * w

    stats = []
    for class_id, label in id2label.items():
        pixels = int((segmentation == class_id).sum())
        if pixels > 0:
            stats.append((class_id, label, 100 * pixels / total))

    return sorted(stats, key=lambda x: x[2], reverse=True)


# ============================================================
# DYNAMIC COLOR MAPPING (NO FIXED COLORS)
# ============================================================


def segmentation_to_rgb(segmentation):
    unique_ids = np.unique(segmentation)
    cmap = cm.get_cmap("tab20", len(unique_ids))

    id_to_color = {
        int(k): (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        for i, k in enumerate(unique_ids)
    }

    h, w = segmentation.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for k, color in id_to_color.items():
        rgb[segmentation == k] = color

    return rgb


# ============================================================
# INTERACTIVE STATS PLOT
# ============================================================


def plot_stats(stats):
    df = pd.DataFrame(stats, columns=["ID", "Label", "Percentage"])
    # Assumiamo che df["Class"] contenga gli stessi ID della segmentazione
    classes = np.unique(df["ID"])

    # Colormap identica a quella della funzione segmentation_to_rgb
    cmap = cm.get_cmap("tab20", len(classes))

    # Generiamo la mappa classe -> HEX (Plotly accetta HEX o stringhe RGB)
    id_to_color = {int(k): to_hex(cmap(i)[:3]) for i, k in enumerate(classes)}
    df["ID_str"] = df["ID"].astype(str)

    fig = px.bar(
        df,
        x="Percentage",
        y="Label",
        orientation="h",
        text="Percentage",
        color="ID_str",  # this tells Plotly to color bars based on ID
        color_discrete_map={
            str(k): v for k, v in id_to_color.items()
        },  # map as strings
    )

    fig.update_layout(
        height=500,
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    return fig


# ============================================================
# APP
# ============================================================


def main():
    st.title("Image Segmentation Demo")

    if not IMAGE_FOLDER.exists():
        st.error(f"Folder not found: {IMAGE_FOLDER}")
        return

    images = load_images(IMAGE_FOLDER)
    if not images:
        st.error("No images found.")
        return

    mode = st.selectbox(
        "Segmentation mode",
        ["Instance (Mask2Former)", "Semantic (SegFormer)"],
    )

    idx = st.slider("Image index", 0, len(images) - 1, 0)
    image = images[idx]["image"]

    if mode.startswith("Instance"):
        processor, model = load_mask2former()
        id2label = model.config.id2label

        with st.spinner("Running instance segmentation…"):
            result = run_instance(image, processor, model)

        segmentation = result["segmentation"].cpu().numpy()
        stats = instance_stats(
            segmentation,
            result["segments_info"],
            id2label,
        )

    else:
        processor, model = load_segformer()
        id2label = model.config.id2label

        with st.spinner("Running semantic segmentation…"):
            segmentation = run_semantic(image, processor, model)

        stats = semantic_stats(segmentation, id2label)

    seg_rgb = segmentation_to_rgb(segmentation)

    col_img, col_seg, col_plot = st.columns([1, 1, 1])

    with col_img:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    with col_seg:
        st.subheader("Segmentation")
        st.image(seg_rgb, use_container_width=True)
    with col_plot:
        st.subheader("Pixel distribution")
        st.plotly_chart(plot_stats(stats), use_container_width=True)


if __name__ == "__main__":
    main()
