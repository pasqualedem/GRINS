import os
import random

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, PILToTensor
from transformers import AutoImageProcessor, AutoModel

from grins.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from grins.data.mit_place_pulse.preprocess import RemoveWatermark
from grins.modeling.regress.dinov3 import DINOv3Linear
from safetensors.torch import load_file
import pandas as pd


st.set_page_config(layout="wide")

# Define the path to the images folder
IMAGE_FOLDER = PROCESSED_DATA_DIR / "svi_Bari_Italy_spacing" / "0"

# Init model
transform = Compose(
    [
        PILToTensor(),
        RemoveWatermark(height=277),
    ]
)
processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov3-vitb16-pretrain-lvd1689m"
)
backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
model = DINOv3Linear(
    backbone, num_tasks=6, num_head_layers=3, activation="GELU", freeze_backbone=True
)
model.load_state_dict(
    load_file(
        PROJ_ROOT
        / "out"
        / "regress"
        / "dinov3"
        / "run1"
        / "checkpoints"
        / "checkpoint_8"
        / "model.safetensors"
    )
)

def score_converter(x):
    # Scores ranges from -4 to +4, we convert them to 1-10 scale
    # First clamp x to be within -4 to +4
    x = max(-4, min(4, x))
    # Then convert to 1-10 scale
    return ((x + 4) / 8) * 9 + 1


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(("jpg", "jpeg", "png")):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append(img)
    return images


def model_predict(images):
    images = [transform(img) for img in images]
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.detach().numpy()


def main():
    st.title("Rivalorizzaziamo Bari attraverso l'Intelligenza Artificiale (GRINS)")
    tasks = ["Vivacità", "Bellezza", "Tristezza", "Noia", "Sicurezza", "Ricchezza"]

    if os.path.isdir(IMAGE_FOLDER):
        images = load_images_from_folder(IMAGE_FOLDER)
        if len(images) >= 5:
            selected_images = random.sample(images, 5)

            # Display selected images full width in five columns
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                col.image(selected_images[idx], use_container_width=True)

            if st.button("Predict"):
                scores = model_predict(selected_images)
                for i in range(6):
                    st.subheader(tasks[i])
                    ranked_indices = np.argsort(scores[:, i])
                    ranked_images = [selected_images[idx] for idx in ranked_indices]
                    ranked_scores = scores[ranked_indices, i]

                    cols = st.columns(5)
                    for idx, col in enumerate(cols):
                        col.image(ranked_images[idx], use_container_width=True)
                        col.caption(f"Punteggio: {'⭐' * (round(score_converter(ranked_scores[idx])))} ({score_converter(ranked_scores[idx]):.2f})")
        else:
            st.warning("Not enough images in the folder. Provide at least 5 images.")
    else:
        st.warning(f"Image folder '{IMAGE_FOLDER}' does not exist.")


if __name__ == "__main__":
    main()
