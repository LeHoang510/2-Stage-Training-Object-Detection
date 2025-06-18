import os
import shutil
import time
import traceback
from pathlib import Path

import torch
from core.logger import Logger
from core.check_hardware import check_hardware
from core.utils import load_yaml, save_yaml, set_seed
from utils.train_helper import get_model, prepare_dataset
from utils.evaluate_helper import evaluate


def test(config_path: Path|str, device: str|torch.device):
    logger = Logger()
    logger.info(f"{'=' * 20} Setting up testing {'=' * 20}")
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = load_yaml(config_path)
    path_config = config.get("path_config", {})
    model_config = config.get("model_config", {})
    dataset_config = config.get("dataset_config", {})
    evaluate_config = config.get("evaluate_config", {})

    model_checkpoint = Path(path_config["checkpoint_path"]) if path_config.get("checkpoint_path") else None
    output_dir = Path(path_config.get("output_dir", "outputs/test/experiment"))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    save_yaml(config, output_dir / config_path.name)
    logger.info(f"Configuration saved to output directory")

    logger.info(f"Loading model from {model_checkpoint}")
    model = get_model(model_config)
    model.to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["model_state_dict"])
    logger.info(f"Model loaded successfully")

    # Prepare dataset
    logger.info("Preparing dataset")
    test_dataset, test_loader = prepare_dataset(dataset_config["test"])
    logger.info(f"Test dataset size: {len(test_dataset)}")

    logger.info(f"{'=' * 20} Starting evaluation {'=' * 20}")
    evaluation_start_time = time.time()
    # Evaluate the model
    test_loss, test_metric = evaluate(model, test_loader, device, metrics=evaluate_config["metrics"])
    evaluation_duration = time.time() - evaluation_start_time

    logger.info(f"Evaluation completed in {evaluation_duration:.2f} seconds")
    logger.info(f"Main metric ({evaluate_config['main_metric']}): {test_metric[evaluate_config['main_metric']]:.4f}")
    logger.info(f"Average loss: {test_loss['total_loss']:.4f}")
    logger.info(f"Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in test_metric.items()])}")
    logger.info(f"Losses: {', '.join([f'{k}: {v:.4f}' for k, v in test_loss.items()])}")

    # Save evaluation results
    results = {
        "evaluation_time": round(evaluation_duration, 2),
        "main_metric": {
            "name": evaluate_config['main_metric'],
            "value": round(test_metric[evaluate_config['main_metric']], 4)
        },
        "metrics": {k: round(v, 4) for k, v in test_metric.items()},
        "losses": {k: round(v, 4) for k, v in test_loss.items()},
    }
    save_yaml(results, output_dir / "evaluation_results.yaml")

    logger.info(f"💾 Evaluation results saved to {output_dir / 'evaluation_results.yaml'}")
    logger.info(f"{'=' * 20} Testing completed {'=' * 20}")

if __name__ == "__main__":
    set_seed(24)  # Set random seed for reproducibility
    logger = Logger("test")
    device, is_cuda = check_hardware(verbose=False)
    config_path = Path("src/configs/mnist/mnist_test_config.yaml")

    try:
        test(config_path, device=device)
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")
        logger.error(traceback.format_exc())
