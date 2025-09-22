import math
from pathlib import Path

import click
import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from hydra.utils import instantiate
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor, get_scheduler
import torchvision.transforms.v2 as T
from torchmetrics.aggregation import MeanMetric

from ...data.mit_place_pulse.dataset import MITPlacePulseDataset
from .model import DINOv3Linear, TaskPairwiseCorrect


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
    tasks_lookup_inv = {v: k for k, v in tasks_lookup.items()}
    num_tasks = len(tasks)
    processor: AutoImageProcessor = config.processor
    backbone: AutoModel = config.backbone
    model = DINOv3Linear(backbone, num_tasks, **config.model_params)
    logger.info("Model instantiated.")

    # Initialize datasets and dataloaders
    transform: Dataset = config.datasets.transform
    train_ds: Dataset = config.datasets.train_partial(transform=transform, processor=processor)
    val_dss: list[Dataset] = [
        config.datasets.val_partial(question=task, transform=transform, processor=processor)
        for task in tasks
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

    # Loss and metrics
    loss_function: nn.Module = config.loss_function_partial(num_tasks=num_tasks)
    correct_function = TaskPairwiseCorrect()
    accuracy_metric = MeanMetric()

    # Optimizer and scheduler
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
    project_config = ProjectConfiguration(
        project_dir=config.project_dir,
        logging_dir=config.project_dir,
        automatic_checkpoint_naming=True
    )
    accelerator: Accelerator = config.accelerator_partial(project_config=project_config)

    # Tracking
    accelerator.init_trackers(
        **config.init_trackers_params,
        config=config_dict,
        init_kwargs={"wandb": {"group": "regress"}},
    )
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
    wandb_tracker.define_metric("train/step")
    wandb_tracker.define_metric("val/step")
    wandb_tracker.define_metric("train/*", step_metric="train/step")
    wandb_tracker.define_metric("val/*", step_metric="val/step")

    # Prepare!
    model, loss_function, accuracy_metric, optimizer, train_dl, scheduler = accelerator.prepare(
        model, loss_function, accuracy_metric, optimizer, train_dl, scheduler
    )
    val_dls = [accelerator.prepare(dl) for dl in val_dls]

    # Training epoch
    train_step = 0
    val_step = 0
    for epoch in range(num_epochs):
        model.train()
        if loss_function.use_task_weighting:
            loss_function.train()
        train_running_loss = 0.0

        # Wrap DataLoader with tqdm
        train_loop = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, batch in enumerate(train_loop):
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
            scores0 = model(pixel_values=images0)  # [B, num_tasks]
            scores1 = model(pixel_values=images1)  # [B, num_tasks]

            # Compute loss
            loss, loss_per_task = loss_function(scores0, scores1, task_idxs, labels)

            # Backward and optimize
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            # Update running loss
            train_running_loss += loss.item()
            train_avg_loss = train_running_loss / (i + 1)

            # Get current learning rate (assuming single param group)
            current_lr = optimizer.param_groups[0]["lr"]

            # Log
            log_loss_per_task = {
                f"train/{tasks_lookup_inv[task_idx]}_batch_loss": task_loss.item()
                for task_idx, task_loss in enumerate(loss_per_task)
            }
            if loss_function.use_task_weighting:
                # log both sigmas and derived weights
                log_weights = {
                    f"train/{tasks_lookup_inv[i]}_sigma": loss_function.sigmas[i].item()
                    for i in range(loss_function.num_tasks)
                }
                log_weights.update({
                    f"train/{tasks_lookup_inv[i]}_weight": (0.5 / (loss_function.sigmas[i] ** 2)).item()
                    for i in range(loss_function.num_tasks)
                })
            else:
                log_weights = {}
                
            log_dict = {
                "train/batch_loss": loss.item(),
                "train/avg_loss": train_avg_loss,
                "train/lr": current_lr,
                "train/step": train_step,
            }
            log_dict = log_dict | log_loss_per_task | log_weights
            accelerator.log(log_dict)
            train_step += 1

            # Update tqdm bar with loss and LR
            train_loop.set_postfix(
                {
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": f"{train_avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

        # Validation
        model.eval()
        loss_function.eval()
        with torch.inference_mode():
            val_step_init = val_step
            for task, task_idx in tasks_lookup.items():
                val_running_loss = 0.0
                val_dl = val_dls[task_idx]
                val_loop = tqdm(
                    val_dl, desc=f"Validation Epoch {epoch+1}/{num_epochs} - {task}", leave=False
                )
                for i, batch in enumerate(val_loop):
                    # Read batch
                    images0, images1, labels, tasks = (
                        batch["images0"],
                        batch["images1"],
                        batch["labels"],
                        batch["questions"],
                    )
                    task_idxs = torch.tensor(
                        [tasks_lookup[t] for t in tasks],
                        dtype=torch.long,
                        device=accelerator.device,
                    )

                    # Forward pass
                    scores0 = model(pixel_values=images0)  # [B, num_tasks]
                    scores1 = model(pixel_values=images1)  # [B, num_tasks]

                    # Compute loss
                    loss, loss_per_task = loss_function(scores0, scores1, task_idxs, labels)

                    # Compute accuracy
                    correct = correct_function(scores0, scores1, task_idxs, labels)
                    batch_accuracy = accuracy_metric(correct)

                    # Update running loss
                    val_running_loss += loss.item()
                    val_avg_loss = val_running_loss / (i + 1)

                    # Log
                    log_dict = {
                        f"val/{task}_batch_loss": loss.item(),
                        f"val/{task}_avg_loss": val_avg_loss,
                        f"val/{task}_batch_accuracy": batch_accuracy,
                        "val/step": val_step,
                    }
                    accelerator.log(log_dict)
                    val_step += 1

                    # Update tqdm bar with loss and LR
                    val_loop.set_postfix(
                        {
                            "batch_loss": f"{loss.item():.4f}",
                            "avg_loss": f"{val_avg_loss:.4f}",
                        }
                    )
                accelerator.log({f"val/{task}_avg_accuracy": accuracy_metric.compute().item()})
                accuracy_metric.reset()
                if task_idx < len(tasks) - 1:
                    val_step = val_step_init
        
        # Save checkpoint
        accelerator.save_state()
        


if __name__ == "__main__":
    cli()
