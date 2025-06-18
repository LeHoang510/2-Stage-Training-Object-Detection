from pathlib import Path
import inspect
import traceback

from torch.utils.data import DataLoader

from core.utils import load_yaml
from core.logger import Logger
from utils.train_helper import DATASETS, MODELS, OPTIMIZERS, SCHEDULERS
from utils.evaluate_helper import EVALUATE_FUNCTIONS


def train_config_validator(config_path: str|Path):
    """
    Validate the training configuration file.

    Args:
        config_path (str | Path): Path to the YAML configuration file.
    Raises:
        ValueError: If any required configuration parameter is missing or invalid.
    Returns:
        None: If the configuration is valid.
    .. warning::
       This function only checks for the existence of the necessary parameters,
       not the data type or value correctness.
       And it also does not give warnings for missing optional parameters.

    """
    logger = Logger()
    config = load_yaml(config_path)

    path_config = config.get("path_config", {})
    model_config = config.get("model_config", {})
    dataset_config = config.get("dataset_config", {})
    train_config = config.get("train_config", {})
    evaluate_config = config.get("evaluate_config", {})

    # Validate path_config
    output_dir = path_config.get("output_dir", None)
    if not output_dir:
        logger.error("[path_config.output_dir] is required")

    for key in ["checkpoint_path", "pretrained_path"]:
        path = path_config.get(key, None)
        if path and not Path(path).exists():
            logger.error(f"[path_config.{key}] is set but path does not exist: {path}")

    # Validate dataset_config
    if "train" not in dataset_config:
        logger.error("[dataset_config] must contain 'train' configurations")
    else:
        train_dataset_config = dataset_config["train"]
        train_dataset_config_errors = _validate_dataset_config(train_dataset_config)
        if train_dataset_config_errors:
            for key, error in train_dataset_config_errors.items():
                logger.error(f"[dataset_config.train.{key}] {error}")

    if "val" in dataset_config:
        val_dataset_config = dataset_config["val"]
        val_dataset_config_errors = _validate_dataset_config(val_dataset_config)
        if val_dataset_config_errors:
            for key, error in val_dataset_config_errors.items():
                logger.error(f"[dataset_config.val.{key}] {error}")
        if "evaluate_config" not in config:
            logger.error("[dataset_config.val] is provided but [evaluate_config] is not set. "
                         "You need to provide [evaluate_config] to evaluate the validation dataset.")

    # Validate model_config
    if "type" not in model_config:
        logger.error("[model_config.type] must exist")
    elif model_config["type"] not in MODELS:
        logger.error(f"[model_config.type] '{model_config['type']}' is not supported. Available options: {list(MODELS.keys())}")
    else:
        model_class = MODELS[model_config["type"]]
        params = model_config.get("params", {})
        params_errors = _validate_params(model_class, params)
        if params_errors:
            logger.error(f"[model_config.params] {model_config['type']} parameters error\n"
                         f"Required params: {params_errors.get('required_params', [])}\n"
                         f"Optional params: {params_errors.get('optional_params', [])}\n"
                         f"Error: {params_errors.get('error', '')}")

    # Validate train_config
    if "num_epochs" not in train_config:
        logger.error("[train_config.num_epochs] must exist")
    elif not isinstance(train_config["num_epochs"], int) or train_config["num_epochs"] <= 0:
        logger.error("[train_config.num_epochs] must be a positive integer")

    if "optimizer" not in train_config:
        logger.error("[train_config.optimizer] must exist")
    elif "type" not in train_config["optimizer"]:
        logger.error("[train_config.optimizer.type] must exist")
    elif train_config["optimizer"]["type"] not in OPTIMIZERS:
        logger.error(f"[train_config.optimizer.type] '{train_config['optimizer']['type']}' is not supported. "
                     f"Available options: {list(OPTIMIZERS.keys())}")

    if "scheduler" in train_config:
        if "type" not in train_config["scheduler"]:
            logger.error("[train_config.scheduler.type] must exist")
        elif train_config["scheduler"]["type"] not in SCHEDULERS:
            logger.error(f"[train_config.scheduler.type] '{train_config['scheduler']['type']}' is not supported. "
                         f"Available options: {list(SCHEDULERS.keys())}")

    # Validate evaluate_config
    if evaluate_config:
        if "metrics" not in evaluate_config:
            logger.error("[evaluate_config.metrics] must exist")
        else:
            for metric in evaluate_config["metrics"]:
                if "function" not in metric:
                    logger.error("[evaluate_config.metrics.function] must exist")
                elif metric["function"] not in EVALUATE_FUNCTIONS:
                    logger.error(f"[evaluate_config.metrics.function] '{metric['function']}' is not supported. "
                                 f"Available options: {list(EVALUATE_FUNCTIONS.keys())}")
        if "main_metric" not in evaluate_config:
            logger.error("[evaluate_config.main_metric] must exist")

    # Check if there are any errors
    if logger.has_errors():
        raise ValueError("Configuration validation failed. Please check the logs for details.")

def test_config_validator(config_path: str|Path):
    """
    Validate the testing configuration file.
    Args:
        config_path (str | Path): Path to the YAML configuration file.
    Raises:
        ValueError: If any required configuration parameter is missing or invalid.
    Returns:
        None: If the configuration is valid.
    .. warning::
       This function only checks for the existence of the necessary parameters,
       not the data type or value correctness.
       And it also does not give warnings for missing optional parameters.

    """
    logger = Logger()
    config = load_yaml(config_path)

    path_config = config.get("path_config", {})
    dataset_config = config.get("dataset_config", {})
    evaluate_config = config.get("evaluate_config", {})
    model_config = config.get("model_config", {})

    # Validate path_config
    output_dir = path_config.get("output_dir", None)
    checkpoint_path = path_config.get("checkpoint_path", None)
    if not output_dir:
        logger.error("[path_config.output_dir] is required")
    if not checkpoint_path:
        logger.error("[path_config.checkpoint_path] is required")
    elif not Path(checkpoint_path).exists():
        logger.error(f"[path_config.checkpoint_path] does not exist: {checkpoint_path}")

    # Validate dataset_config
    if "test" not in dataset_config:
        logger.error("[dataset_config.test] is required")
    else:
        test_dataset_config = dataset_config["test"]
        test_dataset_config_errors = _validate_dataset_config(test_dataset_config)
        if test_dataset_config_errors:
            for key, error in test_dataset_config_errors.items():
                logger.error(f"[dataset_config.test.{key}] {error}")

    # Validate model_config
    if "type" not in model_config:
        logger.error("[model_config.type] is required")
    elif model_config["type"] not in MODELS:
        logger.error(f"[model_config.type] '{model_config['type']}' is not supported. "
                     f"Available options: {list(MODELS.keys())}")
    else:
        model_class = MODELS[model_config["type"]]
        params = model_config.get("params", {})
        params_errors = _validate_params(model_class, params)
        if params_errors:
            logger.error(f"[model_config.params] {model_config['type']} parameters error\n"
                         f"Required params: {params_errors.get('required_params', [])}\n"
                         f"Optional params: {params_errors.get('optional_params', [])}\n"
                         f"Error: {params_errors.get('error', '')}")

    # Validate evaluate_config
    if evaluate_config:
        if "metrics" not in evaluate_config:
            logger.error("[evaluate_config.metrics] is required")
        else:
            for metric in evaluate_config["metrics"]:
                if "function" not in metric:
                    logger.error("[evaluate_config.metrics.function] is required")
                elif metric["function"] not in EVALUATE_FUNCTIONS:
                    logger.error(f"[evaluate_config.metrics.function] '{metric['function']}' is not supported. "
                                 f"Available options: {list(EVALUATE_FUNCTIONS.keys())}")
        if "main_metric" not in evaluate_config:
            logger.error("[evaluate_config.main_metric] is required")

    # Check if there are any errors
    if logger.has_errors():
        raise ValueError("Configuration validation failed. Please check the logs for details.")

def inference_config_validator(config_path: str|Path):
    """
    Validate the inference configuration file.
    Args:
        config_path (str | Path): Path to the YAML configuration file.
    Raises:
        ValueError: If any required configuration parameter is missing or invalid.
    Returns:
        None: If the configuration is valid.
    .. warning::
       This function only checks for the existence of the necessary parameters,
       not the data type or value correctness.
       And it also does not give warnings for missing optional parameters.

    """
    logger = Logger()
    config = load_yaml(config_path)

    path_config = config.get("path_config", {})
    model_config = config.get("model_config", {})
    dataset_config = config.get("dataset_config", {})

    # Validate path_config
    output_dir = path_config.get("output_dir", None)
    checkpoint_path = path_config.get("checkpoint_path", None)
    if not output_dir:
        logger.error("[path_config.output_dir] is required")
    if not checkpoint_path:
        logger.error("[path_config.checkpoint_path] is required")
    elif not Path(checkpoint_path).exists():
        logger.error(f"[path_config.checkpoint_path] does not exist: {checkpoint_path}")

    # Validate dataset_config
    if "inference" not in dataset_config:
        logger.error("[dataset_config.inference] is required")
    else:
        inference_dataset_config = dataset_config["inference"]
        inference_dataset_config_errors = _validate_dataset_config(inference_dataset_config)
        if inference_dataset_config_errors:
            for key, error in inference_dataset_config_errors.items():
                logger.error(f"[dataset_config.inference.{key}] {error}")

    # Validate model_config
    if "type" not in model_config:
        logger.error("[model_config.type] is required")
    elif model_config["type"] not in MODELS:
        logger.error(f"[model_config.type] '{model_config['type']}' is not supported. "
                     f"Available options: {list(MODELS.keys())}")
    else:
        model_class = MODELS[model_config["type"]]
        params = model_config.get("params", {})
        params_errors = _validate_params(model_class, params)
        if params_errors:
            logger.error(f"[model_config.params] {model_config['type']} parameters error\n"
                         f"Required params: {params_errors.get('required_params', [])}\n"
                         f"Optional params: {params_errors.get('optional_params', [])}\n"
                         f"Error: {params_errors.get('error', '')}")

def _validate_dataset_config(x_dataset_config: dict):
    """
    Validate the dataset configuration.
    Args:
        dataset_config (dict): Configuration dictionary for the dataset.
    Returns:
        errors (list): List of error messages if validation fails.
    """
    errors = {}

    if "type" not in x_dataset_config:
        errors["type"] = "must exist"
    elif x_dataset_config["type"] not in DATASETS:
        errors["type"] = f"'{x_dataset_config['type']}' is not supported. Available options: {list(DATASETS.keys())}"
        return errors

    dataset_class = DATASETS[x_dataset_config["type"]]
    params = x_dataset_config.get("dataset_params", {})

    params_errors = _validate_params(dataset_class, params)
    if params_errors:
        errors["dataset_params"] = (
            f"{x_dataset_config['type']} parameters error\n"
            f"Required params: {params_errors.get('required_params', [])}\n"
            f"Optional params: {params_errors.get('optional_params', [])}\n"
            f"Error: {params_errors.get('error', '')}"
        )

    return errors

def _validate_params(cls: type, params: dict):
    """
    Validate the parameters for a given class.

    Args:
        cls (type): The class to validate
        params (dict): The parameters to validate.
    Returns:
        errors (dict): Dictionary of errors if validation fails.
    """
    try:
        sig = inspect.signature(cls.__init__)
        bound_args = sig.bind_partial(**params)
        bound_args.apply_defaults()
    except TypeError as e:
        required_params = []
        optional_params = []

        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.default == inspect.Parameter.empty:
                required_params.append(name)
            else:
                optional_params.append(name)

        return {
            "required_params": required_params,
            "optional_params": optional_params,
            "error": str(e)
        }

    return {}

if __name__ == "__main__":
    logger = Logger("config_validator")
    train_config_path = Path("src/configs/mnist/mnist_train_config.yaml")
    test_config_path = Path("src/configs/mnist/mnist_test_config.yaml")
    inference_config_path = Path("src/configs/mnist/mnist_inference_config.yaml")
    try:
        train_config_validator(train_config_path)
        test_config_validator(test_config_path)
        inference_config_validator(inference_config_path)
        logger.info(f"✅ Configuration validation successful")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        logger.error(traceback.format_exc())
        exit(1)
