from pathlib import Path
import random
import yaml
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int = 24):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def load_yaml(config_path: Path|str):
	"""
    Load a YAML configuration file.
	Args:
		config_path (Path|str): Path to the YAML configuration file.
	Returns:
		dict: Parsed configuration as a dictionary.
	"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config

def load_json(json_path: Path|str):
	"""
    Load a JSON file.
	Args:
		json_path (Path|str): Path to the JSON file.
	Returns:
		dict: Parsed JSON data as a dictionary.
	"""
	with open(json_path, 'r') as f:
		data = json.load(f)
	return data

def save_yaml(data: dict, save_path: Path|str):
    """
    Save a dictionary to a YAML file.
    Args:
        data (dict): Data to save.
        save_path (Path|str): Path where the YAML file will be saved.
    """
    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

def save_json(data: dict, save_path: Path|str):
    """
    Save a dictionary to a JSON file.
    Args:
        data (dict): Data to save.
        save_path (Path|str): Path where the JSON file will be saved.
    """
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

class EarlyStopping:
    """
    A utility class for early stopping during training.
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        mode (str): One of {'min', 'max'}.
            In 'min' mode, training will stop when the quantity monitored has stopped decreasing;
            In 'max' mode, it will stop when the quantity monitored has stopped increasing.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        if mode == 'max':
            self.best_metric = -float('inf')
            self.compare_op = lambda x, y: x > y + min_delta
        elif mode == 'min':
            self.best_metric = float('inf')
            self.compare_op = lambda x, y: x < y - min_delta

    def __call__(self, val_metric: float):
        """
        Call method to update the early stopping state.
        Args:
            val_metric (float): The validation metric to monitor.
        Returns:
            bool: True if an improvement is found, False otherwise.
        """
        if self.compare_op(val_metric, self.best_metric):
            self.best_metric = val_metric
            self.counter = 0
            return True # Improvement found, reset counter
        else:
            self.counter += 1
            return False # No improvement found, increment counter

    def status(self):
        """
        Check if the early stopping condition is met.
        Returns:
            bool: True if the patience limit has been reached, False otherwise.
        """
        return self.counter >= self.patience

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_metric': self.best_metric,
            'mode': self.mode,
        }

    def load_state_dict(self, state_dict: dict):
        self.counter = state_dict['counter']
        self.best_metric = state_dict['best_metric']
        self.mode = state_dict['mode']

class TensorBoard:
    """
    A utility class for writing logs to TensorBoard.
    Args:
		log_dir (Path|str): Directory where TensorBoard logs will be saved.
	"""
    def __init__(self, log_dir: Path|str = Path("outputs/logs")):
        self.log_dir = log_dir
        self.writer = None

    def create_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def write_scalar(self, title: str, value: float, epoch: int):
        """
        Write a single scalar value to TensorBoard.
        Args:
			title (str): The title of the scalar. (e.g. "train/loss")
			value (float): The scalar value to log.
			epoch (int): The epoch number for which the scalar is logged.
		"""
        self.create_writer()
        self.writer.add_scalar(title, value, epoch)

    def write_scalars(self, title: str, values: dict, epoch: int):
        """
        Write multiple scalar values to TensorBoard.
        Args:
			title (str): The title of the scalars. (e.g. "train")
			values (dict): A dictionary of scalar values to log, where keys are the names of the scalars.
			epoch (int): The epoch number for which the scalars are logged.
		"""
        self.create_writer()
        self.writer.add_scalars(title, values, epoch)

    def flush(self):
        """
        Flush the TensorBoard writer to ensure all data is written to disk.
        """
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        """
		Close the TensorBoard writer.
		"""
        if self.writer is not None:
            self.writer.close()
