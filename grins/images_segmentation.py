from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from grins.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):


if __name__ == "__main__":
    app()
