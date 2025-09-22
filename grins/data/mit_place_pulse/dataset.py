import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms


class MITPlacePulseDataset(Dataset):
    """
    PyTorch Dataset for the MIT Place Pulse 2.0 dataset.
    Assumes images are stored in a directory and labels are provided in a CSV file.
    """

    def __init__(
        self,
        data_dir,
        question=None,
        seed=42,
        split="train",
        split_ratio=[0.65, 0.05, 0.30],
        transform=None,
        processor=None,
    ):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            question (str, optional): Specific question to filter the dataset. If None, use all questions.
            seed (int): Random seed for reproducibility.
            split (str): One of 'train', 'val', or 'test' to specify the dataset split.
            split_ratio (list): List of three floats specifying the train, val, test split ratios.
            transform (callable, optional): Optional transform to be applied on a sample.
            processor (AutoImageProcessor, optional): An image processor from HF that might be applied in the collate_fn.
        """
        self.data_dir = Path(data_dir)
        self.question = question
        self.seed = seed
        self.split = split
        self.split_ratio = split_ratio
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
        self.processor = processor
        
        self.images_dir = self.data_dir / "gsv" / "final_photo_dataset"
        self.df = pd.read_csv(self.data_dir / "df.csv")
        
        # get the required split
        train_ratio, val_ratio, test_ratio = split_ratio
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Split ratios must sum to 1."
        train_num_rows = int(len(self.df) * train_ratio)
        val_num_rows = int(len(self.df) * val_ratio)
        train_df, temp_df = train_test_split(self.df, train_size=train_num_rows, random_state=seed)
        val_df, test_df = train_test_split(temp_df, train_size=val_num_rows, random_state=seed)
        if split == "train":
            self.data = train_df.reset_index(drop=True)
        elif split == "val":
            self.data = val_df.reset_index(drop=True)
        elif split == "test":
            self.data = test_df.reset_index(drop=True)
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        # filter by question if specified (the column is "study_question")
        if question is not None:
            self.data = self.data[self.data["study_question"] == question].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_left_name = row["left"]
        img_right_name = row["right"]
        img_left_path = self.images_dir / f"{img_left_name}.jpg"
        img_right_path = self.images_dir / f"{img_right_name}.jpg"

        # Check if image files exist
        if not img_left_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_left_path}")
        if not img_right_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_right_path}")
        image0 = Image.open(img_left_path).convert("RGB")
        image1 = Image.open(img_right_path).convert("RGB")

        choice = row["choice"]
        if choice == "left":
            label = 0
        elif choice == "right":
            label = 1
        else:
            label = -1  # tie or no preference

        if self.transform:
            image0 = self.transform(image0)
            image1 = self.transform(image1)

        sample = {
            "image0": image0,
            "image1": image1,
            "label": label,
            "image0_name": img_left_name,
            "image1_name": img_right_name,
            "question": row["study_question"],
        }
        return sample
    
    def collate_fn(self, batch):
        images0 = torch.stack([item['image0'] for item in batch])
        images1 = torch.stack([item['image1'] for item in batch])
        
        if self.processor is not None:
            images0 = self.processor(images0, return_tensors="pt")["pixel_values"]
            images1 = self.processor(images1, return_tensors="pt")["pixel_values"]
        
        labels = torch.tensor([item['label'] for item in batch])
        image0_names = [item['image0_name'] for item in batch]
        image1_names = [item['image1_name'] for item in batch]
        questions = [item['question'] for item in batch]
        
        return {
            'images0': images0,
            'images1': images1,
            'labels': labels,
            'image0_names': image0_names,
            'image1_names': image1_names,
            'questions': questions
        }


