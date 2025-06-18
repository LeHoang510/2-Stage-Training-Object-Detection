# Getting started

- This guide will help you get started with the project. It covers the basic setup and usage.

## Basic workflow

- The workflow should be like this:
    1. Set up the project by installing the dependencies and configuring the environment, check harward compatibility.
    2. Write your own scripts to preprocess the raw data (for example, process to a json file or csv file for Dataset)
    3. Visualize the data in the notebooks or run the visulization scripts to understand the data better.
    4. Write notebooks code to experiment with the code and test your ideas.
    5. Implement your own Model, Dataset, Transform
    6. Write your own configuration and scripts for training, testing, and inference.
    7. Write your config validator to validate the config file. (optional)
    8. Run the training script to train the model.
    9. Visualize the training progress and other information using TensorBoard.
    10. Run the test script to evaluate the model.
    11. Run the inference script to make predictions with the model.

- Some basic information about the project. This section will help you understand the main points of the project and how to use it effectively
    1. The project is designed to be flexible and easy to use. You can run the training, testing, and inference scripts with minimal configuration.
    2. The project uses a modular structure, where each task (training, testing, inference) has its own script. You can find these scripts in the `src/` directory.
    3. The project uses a configuration file (yaml) for each task, which allows you to easily modify the settings for training, testing, and inference. You can find these configuration files in the `src/configs/` directory. You can refer to the [config.md](config.md) for more details on how to write your own configuration files.
    4. Before implementing your own scripts, you should write it in notebooks first. The notebooks are located in the `notebooks/` directory. They provide a convenient way to experiment with the code and test your ideas before implementing them in the scripts.
    5. You can also refer to the existing scripts in the `src/` directory for examples of how to implement your own scripts. The example scripts are used for mnist classification task.
    6. The project uses a logging system to log the training progress and other information. The logs will be saved in the `logs/` directory. You can refer to the [Logger](#logger) for more details on how to use the logging system.
    7. The project uses a tensorboard to visualize the training progress and other information. You can refer to the [Tensorboard](#tensorboard) for more details on how to use the tensorboard.
    8. All outputs such as model checkpoints, config files, tensorboard logs, predictions, evaluation results, and other outputs will be saved in the `outputs/` directory.
    9. Function in utils directory can be used to help you implement your own scripts. You can skip some initialize step and save time when using these functions for example `prepare_training`. You can find these utility functions in the `src/utils/` directory. These functions are designed to be reusable for simplifying the implementation. Refer to the [Utils](#utils) for more details on how to use these utility functions.

- There will be other classes and rules to follow. Checkout the descriptions below to understand how to use them or just write your own script by following the existing scripts.

## Table of contents

- [Logger](#logger)
- [Tensorboard](#tensorboard)
- [Utils](#utils)
- [Early stopping](#early-stopping)
- [Model](#model)
- [Dataset](#dataset)
- [Config](#config)

## Logger

- The project uses a logging system to log the training progress and other information. The logs will be saved in the `logs/` directory.
- In order to use the logging system, you need to import the `Logger` class from the `src/utils/logger.py` file. You can then create an instance of the `Logger` class and use it to log messages.

    ````python
    from src.utils.logger import Logger
    logger = Logger("the_directory_name")
    ````

- After the first initialization, you will have a single instance of the `Logger` class that you can use throughout your script, just import and then `logger = Logger()` is enough to reuse it.

- There are several methods available in the `Logger` class to log messages:

    ```python
    from src.utils.logger import Logger
    logger = Logger()

    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    logger.critical("This is a critical message")
    ```

## Tensorboard

- The project uses TensorBoard to visualize the training progress and other information. You can run TensorBoard by using the command below:

    ```bash
    tensorboard --logdir=<path_to_tensorboard_logs>
    ```

- You can use `TensorBoard` by importing from the `core.utils.tensorboard` module. You can then create an instance of the `TensorBoard` class and use it to log messages.

    ```python
    from src.utils.tensorboard import TensorBoard
    tensorboard = TensorBoard("output_path_here")
    ```

- To log scalar values, you can use the `add_scalar` method:

    ```python
    tensorboard.add_scalar("loss", loss_value, step)
    tensorboard.add_scalar("accuracy", accuracy_value, step)
    ```

- To log scalars, you can use the `add_scalars` method:

    ```python
    tensorboard.add_scalars("loss", {"train": train_loss, "val": val_loss}, step)
    ```

- Normally, you already have tensorboard automatically when using `prepare_training` function from `train_helpers.py`.

## Utils

- In order to use these utility, make sure to define the class/function you want to use in the dictionaries.

- for example, if you want to use VOCModel that you write in `src/models/voc_model.py`, you should add it to `MODELS` dict in `train_helper` like this:

    ```python
    from src.models.voc_model import VOCModel

    models = {
        "MnistDataset": MnistDataset,
        "VOCModel": VOCModel,
        # other models...
    }
    ```

- The same thing with others like `DATASETS`, `OPTIMIZERS`, `SCHEDULER`, etc.
- You can also add your own utility in the same way

## Early stopping

- You can use the `EarlyStopping` class to stop the training process if the validation loss does not improve for a certain number of epochs. This can help prevent overfitting and save training time.
- You can import the `EarlyStopping` class from the `src/utils/early_stopping.py` file and create an instance of it:

    ```python
    from core.utils.early_stopping import EarlyStopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, mode='min')
    ```

- **patience**: The number of epochs to wait for improvement before stopping the training.
- **min_delta**: The minimum change in the monitored quantity to qualify as an improvement.
- **mode**: The mode to use for monitoring the quantity. It can be either 'min' or 'max'. If 'min', training will stop when the quantity stops decreasing. If 'max', training will stop when the quantity stops increasing.

- Normally, you already have early stopping automatically when using `prepare_training` function from `train_helpers.py`.

## Model

- You can write your model as you want (inherit from nn.Module of course), but it is recommended to return predictions and a dictionary of losses in the `forward` method. This will make it easier to use the model in the training and testing scripts.

## Dataset

- Write your own dataset class by inheriting from `Dataset` class in `src/datasets/dataset.py`.
- the dataset should have:
    1. `__init__` method to initialize the dataset.
    2. `__len__` method to return the length of the dataset.
    3. `__getitem__` method to return a single item from the dataset.
    4. `collate_fn` method to collate the items into a batch (this is a static method). This will be used by the DataLoader to collate the items into a batch.

- `__getitem__` and `collate_fn` methods should return a dictionary depending on what you want to use.

## Config

- The project uses a configuration file (yaml) for each task. You can use the `load_yaml` function from `core.utils` to load the configuration file as json and then use it in your sript.

    ```python
    from src.utils import load_yaml
    config = load_yaml("path/to/config.yaml")
    value = config.get("key", "default_value")
    ```

- You can also pass the configuration as parameters using **kwargs. So all you have to do is to define exactly the same keys in the config file as the parameters in your function

    ```python
    from src.utils import load_yaml
    config = load_yaml("path/to/config.yaml")
    params = config.get("params", {})
    my_function(**params)
    ```
