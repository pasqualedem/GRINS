from pathlib import Path
import pandas as pd
import os
from ...config import EXTERNAL_DATA_DIR
from torchvision.transforms.v2 import Transform


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


class RemoveWatermark(Transform):
    def __init__(self, height: int = 277):
        super().__init__()
        self.height = height

    def transform(self, img, params):
        return img[:, : self.height, :]

    def __repr__(self):
        return f"{self.__class__.__name__}(height={self.height})"


if __name__ == "__main__":
    remove_rows_with_missing_images(EXTERNAL_DATA_DIR / "mit-place-pulse")
