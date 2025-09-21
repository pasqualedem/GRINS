import math
from pathlib import Path

import click
import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from hydra.utils import instantiate
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, get_scheduler
import torchvision.transforms.v2 as T

from ...data.mit_place_pulse.dataset import MITPlacePulseDataset
from .model import DINOv3Linear


@click.group
def cli():
    pass


@cli.command
@click.option("--config-path", type=click.Path(exists=True), required=True)
def train(config_path: Path | str):
    # read the yaml
    config_path = Path(config_path)
    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f)
    config = instantiate(config_dict)

    # Set seed
    set_seed(config.seed)

    # Initialize model
    tasks: list[str] = config.tasks
    tasks_lookup = {task: i for i, task in enumerate(tasks)}
    num_tasks = len(tasks)
    processor: AutoProcessor = config.backbone
    backbone: AutoModel = config.backbone
    model = DINOv3Linear(backbone, num_tasks, **config.model_params)
    logger.info("Model instantiated.")

    # Initialize datasets and dataloaders
    transform: Dataset = config.datasets.transform
    transform = T.Compose([transform, T.Lambda(lambda x: processor(x, return_tensors="pt"))])
    train_ds: Dataset = config.datasets.train_partial(transform=transform)
    val_dss: list[Dataset] = [
        config.datasets.val_partial(question=task, transform=transform) for task in tasks
    ]
    train_dl: DataLoader = config.dataloaders.train_partial(
        dataset=train_ds, collate_fn=train_ds.collate_fn
    )
    val_dls: list[DataLoader] = [
        config.dataloaders.val_partial(dataset=ds, collate_fn=ds.collate_fn) for ds in val_dss
    ]
    logger.info("Datasets instantiated.")
    logger.info(f"Train dataset size: {len(train_ds)}")
    for ds in val_dss:
        logger.info(f"Val dataset size for {ds.question}: {len(ds)}")

    # Other training things
    num_epochs = config.num_epochs
    num_epoch_steps = len(train_dl)
    num_training_steps = num_epoch_steps * num_epochs
    num_warmup_steps = math.ceil(num_epoch_steps * config.warmup_ratio)

    # Loss, optimizer and scheduler
    loss_function: nn.Module = config.loss_function_partial(num_tasks=num_tasks)
    params_to_optimize = list(model.parameters())
    if loss_function.use_task_weighting:
        params_to_optimize += list(loss_function.parameters())
    optimizer = config.optimizer_partial(params_to_optimize)
    scheduler = get_scheduler(
        config.scheduler_name, optimizer, num_warmup_steps, num_training_steps
    )
    logger.info(
        f"Scheduler {config.scheduler_name} initialized with {num_warmup_steps} warmup steps."
    )

    # Accelerate!
    accelerator: Accelerator = config.accelerator
    model, loss_function, optimizer, train_dl, scheduler = accelerator.prepare(
        model, loss_function, optimizer, train_dl, scheduler
    )
    val_dls = [accelerator.prepare(dl) for dl in val_dls]

    # Training epoch
    for epoch in range(num_epochs):
        model.train()
        if loss_function.use_task_weighting:
            loss_function.train()
        running_loss = 0.0

        # Wrap DataLoader with tqdm
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, batch in enumerate(loop):
            optimizer.zero_grad()

            # Read batch
            images0, images1, labels, tasks = (
                batch["images0"],
                batch["images1"],
                batch["labels"],
                batch["questions"],
            )
            task_idxs = torch.tensor(
                [tasks_lookup[t] for t in tasks], dtype=torch.long, device=accelerator.device
            )

            # Forward pass
            scores0 = model(images0)  # [B, num_tasks]
            scores1 = model(images1)  # [B, num_tasks]

            # Compute loss
            loss = loss_function(scores0, scores1, task_idxs, labels)

            # Backward and optimize
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            # Update running loss
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)

            # Get current learning rate (assuming single param group)
            current_lr = optimizer.param_groups[0]["lr"]

            # Update tqdm bar with loss and LR
            loop.set_postfix(
                {
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )


if __name__ == "__main__":
    cli()
