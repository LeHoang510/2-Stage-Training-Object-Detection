import torch
from tqdm import tqdm

from evaluate.mnist_evaluate import mnist_accuracy, mnist_f1_score

EVALUATE_FUNCTIONS = {
    "mnist_accuracy": mnist_accuracy,
    "mnist_f1_score": mnist_f1_score,
}

@torch.no_grad()
def evaluate(model, dataloader, device, metrics={}):
    """
    Evaluate the model on the given dataloader using specified metrics.
    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (str|torch.device): Device to run the evaluation on (e.g., 'cuda' or 'cpu').
        metrics (list): List of metrics to calculate. Each metric should be a dict with 'function' and optional 'params'.
    Returns:
        tuple: Average loss (dict) and metric results (dict).
    """
    for metric in metrics:
        metric_fn = metric.get("function")
        if metric_fn not in EVALUATE_FUNCTIONS:
            raise ValueError(f"Unsupported metric function: {metric_fn}. Available options: {list(EVALUATE_FUNCTIONS.keys())}")

    model.eval()
    all_preds = []
    all_targets = []
    all_losses = {}
    metric_results = {}

    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Evaluating"):
            images = sample["images"].to(device)
            targets = sample["targets"].to(device)

            predictions, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            all_preds.extend(predictions)
            all_targets.extend(targets)

            all_losses["total_loss"] = all_losses.get("total_loss", 0.0) + losses.item()
            for k, v in loss_dict.items():
                all_losses[k] = all_losses.get(k, 0.0) + v.item()

    # Calculate average loss and metric
    avg_all_losses = {k: v / len(dataloader) for k, v in all_losses.items()}

    # Calculate metrics
    for metric in metrics:
        metric_fn = metric.get("function")
        metric_results.update(EVALUATE_FUNCTIONS[metric_fn](all_preds, all_targets, **metric.get("params", {})))

    return avg_all_losses, metric_results
