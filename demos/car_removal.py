import streamlit as st
from pathlib import Path
import re
from collections import defaultdict
from PIL import Image

ROOT = Path("data/raw/svi/Bari_Italy/car_removal")

FILENAME_RE = re.compile(
    r"image_(\d+_\d+)_heading_(\d+)_removal_(\d+)\.png"
)

@st.cache_data
def index_dataset(root: Path):
    data = {}

    for heading_dir in root.iterdir():
        if not heading_dir.is_dir() or not heading_dir.name.isdigit():
            continue

        heading = heading_dir.name
        if heading not in data:
            data[heading] = {}

        for img_path in heading_dir.glob("*.png"):
            match = FILENAME_RE.match(img_path.name)
            if not match:
                continue

            scene_id, fname_heading, removal_idx = match.groups()
            removal_idx = int(removal_idx)

            # sanity check
            if fname_heading != heading:
                st.warning(f"Heading mismatch: {img_path.name}")

            if scene_id not in data[heading]:
                data[heading][scene_id] = {}

            data[heading][scene_id][removal_idx] = img_path

    return data



data = index_dataset(ROOT)

st.title("Car Removal Visualization")

# ---- Controls ----
heading = st.selectbox(
    "Heading (degrees)",
    sorted(data.keys(), key=int)
)

scenes = sorted(data[heading].keys())
scene_id = st.selectbox(
    "Scene",
    scenes
)

versions = data[heading][scene_id]
max_version = max(versions.keys())

version = st.slider(
    "Removal step",
    min_value=0,
    max_value=max_version,
    value=0,
    step=1
)

# ---- Display ----
# ---- Display controls ----
show_before_after = st.checkbox(
    "Before / After comparison",
    value=False
)

# ---- Display ----
if version not in versions:
    st.error("Selected removal step does not exist.")
else:
    after_img = Image.open(versions[version])

    if show_before_after:
        if 0 not in versions:
            st.error("removal_0 not found: cannot show 'Before' image.")
        else:
            before_img = Image.open(versions[0])

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    before_img,
                    caption="Before (removal_0)",
                    use_container_width=True
                )

            with col2:
                st.image(
                    after_img,
                    caption=f"After (removal_{version})",
                    use_container_width=True
                )
    else:
        st.image(
            after_img,
            caption=f"Scene {scene_id} | Heading {heading}° | Removal {version}",
            use_container_width=True
        )
