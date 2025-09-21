from pathlib import Path
import pandas as pd
import os
from ...config import EXTERNAL_DATA_DIR


def remove_rows_with_missing_images(data_dir: Path):
    data_dir = Path(data_dir)
    images_dir = data_dir / "gsv" / "final_photo_dataset"
    df = pd.read_csv(data_dir / "votes_clean.csv")

    def image_exists(image_id):
        image_path = images_dir / f"{image_id}.jpg"
        return image_path.exists()

    # check if both images in column "left" and "right" exist
    mask = df["left"].apply(image_exists) & df["right"].apply(image_exists)
    cleaned_df = df[mask]
    cleaned_df.to_csv(data_dir / "df.csv", index=False)


if __name__ == "__main__":
    remove_rows_with_missing_images(
        EXTERNAL_DATA_DIR / "mit-place-pulse"
    )
