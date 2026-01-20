# GRINS

## Setup

### 1. Create and activate a UV environment

```bash
# Create a new UV environment
uv sync

# Activate the environment
source .venv/bin/activate
```

## Data Preparation

### 1. MIT Place Pulse 2.0 Dataset

1. Download the [MIT Place Pulse 2.0 dataset](https://www.kaggle.com/datasets/shubham6147/mit-place-pulse) from Kaggle
   
2. Unzip the dataset into the `data/external` directory:
    ```bash
    # Create the data directory if it doesn't exist
    mkdir -p data/external

    # Unzip the downloaded dataset
    unzip ~/Downloads/mit-place-pulse.zip -d data/external/
    ```

3. Preprocess the dataset:
    ```bash
    python -m grins.data.mit_place_pulse.preprocess
    ```
    This will create a `df.csv` file in the data directory, with cleaned annotations (e.g. only rows with valid images).

### 2. Custom SVI Dataset

1. Download coordinates and generate visualization:
    ```bash
    python -m grins.data.svi.download_coordinates -l "Bari, Italy" -m spacing -n 10000 -s 40 -o 15 -z 2500
    ```

2. Download street view images for the coordinates:
    ```bash
    python -m grins.data.svi.download_images -l "Bari, Italy" -m spacing
    ```

## Cars Removal

To remove cars from street view images, run the following command to obtain the annotated images:

```bash
python -m grins.modeling.cars_removal.detect annotate data/raw/svi/Bari_Italy --batch-size 4
```

Then, to inpaint the images and remove the cars, run:

```bash
python -m grins.modeling.cars_removal.remove data/raw/svi/Bari_Italy
```

## Run a Training Experiment

To run a training experiment, use the following command:

```bash
accelerate launch --num-processes=1 -m grins.modeling.regress.train train --config-path=models/configs/regress/dinov3_multilayer_head.yaml
```