import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def mnist_accuracy(predictions, targets):
    """
    Calculate accuracy for MNIST classification.
    
    Args:
        predictions: List of model outputs (log probabilities from log_softmax)
        targets: List of target labels
        
    Returns:
        float: Accuracy percentage (0-100)
    """
    # Convert list of predictions to tensor
    preds_tensor = torch.stack(predictions)
    targets_tensor = torch.stack(targets)
    
    # Get predicted classes (argmax of log probabilities)
    _, pred_classes = preds_tensor.max(dim=1)
    
    # Calculate accuracy
    correct = (pred_classes == targets_tensor).sum().item()
    total = targets_tensor.size(0)
    accuracy = 100.0 * correct / total

    return {"accuracy": accuracy}

def mnist_f1_score(predictions, targets, average='weighted'):
    """
    Calculate F1 score for MNIST classification.
    
    Args:
        predictions: List of model outputs (log probabilities from log_softmax)
        targets: List of target labels
        
    Returns:
        float: F1 score
    """
    
    # Convert list of predictions to tensor
    preds_tensor = torch.stack(predictions)
    targets_tensor = torch.stack(targets)
    
    # Get predicted classes (argmax of log probabilities)
    _, pred_classes = preds_tensor.max(dim=1)
    
    # Calculate F1 score
    f1 = f1_score(targets_tensor.cpu(), pred_classes.cpu(), average=average)

    return {"f1_score": f1}