import os
import random
import datetime

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, PILToTensor
from transformers import AutoImageProcessor, AutoModel

from grins.config import DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from grins.data.mit_place_pulse.preprocess import RemoveWatermark
from grins.modeling.regress.dinov3 import DINOv3Linear
from safetensors.torch import load_file
import pandas as pd


TASKS = ["Vivacità", "Bellezza", "Tristezza", "Noia", "Sicurezza", "Ricchezza"]

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
            images.append({"image": img, "filename": filename})
    return images


def model_predict(images):
    images = [transform(img) for img in images]
    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.detach().numpy()


def init_selected_images(images):
    if "selected_images" not in st.session_state:
        st.session_state.selected_images = random.sample(images, 5)
        

def clear_rank_keys():
    # rimuove chiavi di ranking in session_state quando si rimescolano le immagini
    for task in TASKS:
        for idx in range(5):
            key = f"{task}_rank_{idx}"
            if key in st.session_state:
                del st.session_state[key]

def save_results_to_csv(user_rankings, selected_images):
    # Salva i risultati in un file CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    for task in TASKS:
        ranks = user_rankings[task]
        for idx, rank in enumerate(ranks):
            results.append({
                "timestamp": timestamp,
                "task": task,
                "image_filename": selected_images[idx]["filename"],
                "user_rank": rank
            })
    df = pd.DataFrame(results)
    output_file = DATA_DIR / "demo_results.csv"
    if output_file.exists():
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, index=False)


def main():
    st.title("Rivalorizziamo Bari attraverso l'Intelligenza Artificiale (GRINS) — Ranking utente vs AI")
    if not os.path.isdir(IMAGE_FOLDER):
        st.warning(f"La cartella immagini '{IMAGE_FOLDER}' non esiste.")
        return

    images = load_images_from_folder(IMAGE_FOLDER)
    if len(images) < 5:
        st.warning("Not enough images in the folder. Provide at least 5 images.")
        return

    # inizializza immagini selezionate una sola volta per sessione
    init_selected_images(images)
    if st.button("Nuove immagini (rimescola)"):
        st.session_state.selected_images = random.sample(images, 5)
        clear_rank_keys()

    selected_images = st.session_state.selected_images  # lista di percorsi

    # Mostra le 5 immagini in alto, con caption (nome file)
    st.subheader("Immagini da valutare (1 = migliore)")
    cols = st.columns(5)
    for idx, img in enumerate(selected_images):
        try:
            cols[idx].image(img["image"], use_container_width=True)
            cols[idx].caption(img["filename"])
        except Exception:
            cols[idx].text("Errore immagine")

    st.markdown("---")
    st.write("Per ogni criterio assegna **un rango unico** da 1 a 5 a ciascuna immagine (1 = migliore).")

    # Form che raccoglie tutti i ranghi; submit quando l'utente ha finito
    with st.form("ranking_form"):
        task_cols = st.columns(len(TASKS) // 2)
        for task_i, task in enumerate(TASKS):
            with task_cols[task_i % (len(TASKS) // 2)]:
                st.markdown(f"#### {task}")
                cols = st.columns(5)
                for idx, col in enumerate(cols):
                    # opzioni 1..5, valore di default = posizione attuale (più naturale)
                    default = max(1, min(5, idx + 1))
                    col.selectbox(
                        label="",
                        options=[1, 2, 3, 4, 5],
                        index=default - 1,
                        key=f"{task}_rank_{idx}",
                    )
        submitted = st.form_submit_button("Invia ranking e confronta con AI")

    if submitted:
        # Validazione: per ogni task i ranghi devono essere unici
        user_rankings = {}  # task -> list di 5 rank (index = image index)
        valid = True
        for task in TASKS:
            ranks = [st.session_state.get(f"{task}_rank_{idx}", None) for idx in range(5)]
            if any(r is None for r in ranks):
                st.error(f"Compila tutti i ranghi per '{task}'.")
                valid = False
            elif len(set(ranks)) != 5:
                st.error(f"Per '{task}' ci sono ranghi duplicati. Usa i numeri 1..5 senza ripetizioni.")
                valid = False
            else:
                user_rankings[task] = ranks

        if not valid:
            st.warning("Correggi gli errori sopra e reinvia.")
            return
        
        # Salva i risultati in CSV
        save_results_to_csv(user_rankings, selected_images)

        # Ottieni punteggi AI
        ai_scores = model_predict([elem["image"] for elem in selected_images])  # shape: (5, len(TASKS))
        if ai_scores.shape != (5, len(TASKS)):
            st.error(f"model_predict deve restituire array shape (5, {len(TASKS)}). Hai {ai_scores.shape}.")
            return

        # Confronto per ogni criterio
        corrs = []
        for i, task in enumerate(TASKS):
            st.subheader(task)

            # --- Ranking utente (ordina indici per rank crescente, 1 = migliore)
            ranks = np.array(user_rankings[task])         # index -> rank (1..5)
            user_order = np.argsort(ranks)               # array di indici immagini (best->worst)
            st.write("**Ranking utente** (1 = migliore):")
            cols = st.columns(5)
            for pos, c in enumerate(cols):
                img_idx = user_order[pos]
                try:
                    c.image(selected_images[img_idx]["image"], use_container_width=True)
                except Exception:
                    c.text("Errore immagine")
                c.caption(f"Rank utente: {ranks[img_idx]}")

            # --- Ranking AI (ordina secondo punteggio; assumiamo 'più alto = più presenza del criterio')
            ai_order = np.argsort(ai_scores[:, i])[::-1]  # best->worst
            st.write("**Ranking AI** (punteggio grezzo):")
            cols = st.columns(5)
            for pos, c in enumerate(cols):
                img_idx = ai_order[pos]
                try:
                    c.image(selected_images[img_idx]["image"], use_container_width=True)
                except Exception:
                    c.text("Errore immagine")
                c.caption(f"Punteggio AI: {score_converter(ai_scores[img_idx, i]):.2f}")

            # --- Correlazione tipo Spearman (Pearson sulle ranks)
            # Per ogni immagine costruisco il rank: user_rank(img_idx) e ai_rank(img_idx)
            ai_ranks_per_img = np.empty(5, dtype=int)
            for img_idx in range(5):
                ai_ranks_per_img[img_idx] = int(np.where(ai_order == img_idx)[0][0]) + 1  # 1..5
            user_ranks_per_img = ranks  # 1..5, index = immagine

            # trasformo i ranghi in "score" dove 5 = migliore (6 - rank)
            user_score_like = 6 - user_ranks_per_img
            ai_score_like = 6 - ai_ranks_per_img

            if np.std(user_score_like) == 0 or np.std(ai_score_like) == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(user_score_like, ai_score_like)[0, 1]

            corrs.append(corr if not np.isnan(corr) else 0.0)
            if np.isnan(corr):
                st.warning("Impossibile calcolare correlazione (deviazione nulla).")
            else:
                st.write(f"**Correlazione (approssimazione Spearman)** tra utente e AI: {corr:.2f}")

        # media delle correlazioni (ignora NaN)
        valid_corrs = [c for c in corrs if not np.isnan(c)]
        if valid_corrs:
            avg_corr = float(np.mean(valid_corrs))
            st.markdown("---")
            st.write(f"**Accordo medio tra utente e AI (media sui criteri): {avg_corr:.2f}**")


if __name__ == "__main__":
    main()
