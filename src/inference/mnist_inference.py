import os
import random
import shutil
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.logger import Logger
from core.check_hardware import check_hardware
from core.utils import load_yaml, save_yaml, set_seed
from utils.train_helper import get_model, prepare_dataset
from utils.visualize_helper import plot_gray_image, denormalize
from utils.evaluate_helper import EVALUATE_FUNCTIONS

@torch.no_grad()
def inference_all(model: nn.Module,
                  dataloader: DataLoader,
                  device: str|torch.device,
                  config: dict):
    """
    Function to handle inference for all data in the dataset.
    This function should iterate through the dataloader and perform inference on each batch.
    """
    logger = Logger()

    path_config = config.get("path_config")
    output_dir = Path(path_config.get("output_dir", "outputs/inference/experiment"))

    inference_config = config.get("inference_config", {})
    display = inference_config.get("display", False)
    save = inference_config.get("save", False)
    log = inference_config.get("log", True)

    evaluate_config = config.get("evaluate_config", {})
    metrics = evaluate_config.get("metrics", [])

    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    if save:
        image_dir = output_dir/"predictions"
        os.makedirs(image_dir, exist_ok=True)
        logger.info(f"Saving predictions to {image_dir}")

    model.eval()
    all_preds = []
    all_targets = []
    all_losses = {}
    metric_results = {}

    with torch.no_grad():
        for sample in dataloader:
            images = sample["images"].to(device)
            targets = sample["targets"].to(device)
            ids = sample["ids"]

            predictions, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            all_preds.extend(predictions)
            all_targets.extend(targets)

            all_losses["total_loss"] = all_losses.get("total_loss", 0.0) + losses.item()
            for k, v in loss_dict.items():
                all_losses[k] = all_losses.get(k, 0.0) + v.item()

            probs = torch.exp(predictions)
            max_probs, predicted_classes = torch.max(probs, dim=1)

            if log or display or save:
                for i, (image, target) in enumerate(zip(images, targets)):
                    id = ids[i]
                    label = target.cpu().item()
                    pred = f"{predicted_classes[i].cpu().item()}_{max_probs[i].cpu().item():.2f}"

                    if log:
                        logger.info(f"[Processing image id {id}] Label: {label}, Predicted: {pred}")

                    if display or save:
                        title = f"id_{id}/label_{label}/pred_{pred}"
                        image = denormalize(image, mean=(0.1307,), std=(0.3081,)).cpu()
                        save_path = image_dir / f"image_{i}_pred_{pred}.png" if save else None
                        plot_gray_image(image=image,
                                        title=title,
                                        display=display,
                                        save_path=save_path)

    avg_all_losses = {k: v / len(dataloader) for k, v in all_losses.items()}

    if metrics:
        # Calculate metrics
        for metric in metrics:
            metric_fn = metric.get("function")
            metric_results.update(EVALUATE_FUNCTIONS[metric_fn](all_preds, all_targets, **metric.get("params", {})))
        logger.info(f"Metrics calculated: {', '.join([f'{k}: {v:.4f}' for k, v in metric_results.items()])}")
        logger.info(f"Losses: {', '.join([f'{k}: {v:.4f}' for k, v in avg_all_losses.items()])}")

        results = {
            "Losses": avg_all_losses,
            "Metrics": metric_results,
        }
        save_yaml(results, output_dir / "inference_results.yaml")

def inference_single(model: nn.Module,
                     dataset: Dataset,
                     device: str|torch.device,
                     config: dict):
    """
    Function to handle inference for a single data point.
    This function should take a single image and perform inference on it.
    """
    logger = Logger()

    path_config = config.get("path_config")
    output_dir = Path(path_config.get("output_dir", "outputs/inference/experiment"))

    inference_config = config.get("inference_config", {})
    display = inference_config.get("display", False)
    save = inference_config.get("save", False)
    log = inference_config.get("log", True)
    id = inference_config.get("id", 0)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    if save:
        image_dir = output_dir/"predictions"
        os.makedirs(image_dir, exist_ok=True)
        logger.info(f"Saving predictions to {image_dir}")

    model.eval()

    with torch.no_grad():

        sample = dataset[id]
        image, target = sample["image"], sample["target"]
        image = image.to(device)

        predictions, _ = model(image.unsqueeze(0))
        probs = torch.exp(predictions)
        max_probs, predicted_classes = torch.max(probs, dim=1)

        if log or display or save:

            if log:
                logger.info(f"[Processing single image] Label: {target}, Predicted: {predicted_classes[0].cpu().item()}_{max_probs[0].cpu().item():.2f}")

            if display or save:
                title = f"id_{sample['id']}/label_{target}/pred_{predicted_classes[0].cpu().item()}_{max_probs[0].cpu().item():.2f}"
                image = denormalize(image, mean=(0.1307,), std=(0.3081,)).cpu()
                save_path = image_dir / "single_image_prediction.png" if save else None
                plot_gray_image(image=image,
                                title=title,
                                display=display,
                                save_path=save_path)


def inference_random(model: nn.Module,
                     dataset: Dataset,
                     device: str|torch.device,
                     config: dict):
    """
    Function to handle inference for a random data point.
    This function should randomly select an image from the dataset and perform inference on it.
    """
    logger = Logger()

    path_config = config.get("path_config")
    output_dir = Path(path_config.get("output_dir", "outputs/inference/experiment"))

    inference_config = config.get("inference_config", {})
    display = inference_config.get("display", False)
    save = inference_config.get("save", False)
    log = inference_config.get("log", True)
    id = random.randint(0, len(dataset) - 1)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    if save:
        image_dir = output_dir/"predictions"
        os.makedirs(image_dir, exist_ok=True)
        logger.info(f"Saving predictions to {image_dir}")

    model.eval()

    with torch.no_grad():

        sample = dataset[id]
        image, target = sample["image"], sample["target"]
        image = image.to(device)

        predictions, _ = model(image.unsqueeze(0))
        probs = torch.exp(predictions)
        max_probs, predicted_classes = torch.max(probs, dim=1)

        if log or display or save:
            if log:
                logger.info(f"[Processing single image] Label: {target}, Predicted: {predicted_classes[0].cpu().item()}_{max_probs[0].cpu().item():.2f}")

            if display or save:
                title = f"id_{sample['id']}/label_{target}/pred_{predicted_classes[0].cpu().item()}_{max_probs[0].cpu().item():.2f}"
                image = denormalize(image, mean=(0.1307,), std=(0.3081,)).cpu()
                save_path = image_dir / "single_image_prediction.png" if save else None
                plot_gray_image(image=image,
                                title=title,
                                display=display,
                                save_path=save_path)

def inference(config_path: str|Path, device: str|torch.device):
    """
    Main inference function to run a model on a dataset based on the provided configuration.
    Args:
        config_path (str|Path): Path to the YAML configuration file.
        device (str|torch.device): Device to run the inference on (e.g., 'cuda' or 'cpu').
    """
    logger = Logger()
    logger.info(f"{'=' * 20} Setting up inference {'=' * 20}")
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = load_yaml(config_path)
    path_config = config.get("path_config", {})
    model_config = config.get("model_config", {})
    dataset_config = config.get("dataset_config", {})
    inference_config = config.get("inference_config", {})

    model_checkpoint = Path(path_config["checkpoint_path"]) if path_config.get("checkpoint_path") else None

    logger.info(f"Loading model from {model_checkpoint}")
    model = get_model(model_config)
    model.to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["model_state_dict"])
    logger.info(f"Model loaded successfully")

    # Prepare dataset
    logger.info("Preparing dataset")
    inference_dataset, inference_loader = prepare_dataset(dataset_config["inference"])
    logger.info(f"Inference dataset size: {len(inference_dataset)}")

    logger.info(f"{'=' * 20} Starting inference {'=' * 20}")
    inference_start_time = time.time()

    mode = inference_config.get("mode", "all")
    if mode == "all":
        inference_all(model, inference_loader, device, config)
    elif mode == "single":
        inference_single(model, inference_dataset, device, config)
    elif mode == "random":
        inference_random(model, inference_dataset, device, config)


    inference_end_time = time.time()
    logger.info(f"🕒 Inference completed in {inference_end_time - inference_start_time:.2f} seconds")
    logger.info(f"{'=' * 20} Inference finished {'=' * 20}")


if __name__ == "__main__":
    set_seed(24)  # Set a fixed seed for reproducibility
    logger = Logger("inference")
    device, is_cuda = check_hardware(verbose=False)
    config_path = Path("src/configs/mnist/mnist_inference_config.yaml")

    try:
        inference(config_path, device=device)
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        logger.error(traceback.format_exc())
