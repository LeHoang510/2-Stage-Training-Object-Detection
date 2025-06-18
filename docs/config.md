# Yaml configuration files

- Remmember that the yaml configuration depend on how you want to implement your script. The configuration below is just an example that can be used for this project.

## Table of contents

- [Training Configuration](#training-configuration-configstrain_configyaml)
- [Test Configuration](#test-configuration-configstest_configyaml)
- [Inference Configuration](#inference-configuration-configsinference_configyaml)

## Training Configuration (**configs/train_config.yaml**)

### train_path_config

- path_config including the output directory, checkpoint path, and pretrained path.
- The `output_dir` is required, and if not specified, it will save in the default directory.
- The `checkpoint_path` and `pretrained_path` are optional (can be null or not specified). If specified, it will load the checkpoint or pretrained model from the given path.

    ```yaml
    path_config:
        output_dir: outputs/train/mnist_experiment # Required or will be saved in the default
        # checkpoint_path: null   # Optional
        # pretrained_path: null   # Optional
    ```

### train_dataset_config

- dataset_config including the training and validation datasets.
- train dataset is required, and if not specified, it will raise an error.
- val dataset is optional, and if not specified, validation will not be performed.
- Each dataset has a type, dataset_params, and dataloader_params.
- The `type` is the dataset class name, which is required (should be add to the dictionary in `train_helper.py`).
- The `dataset_params` is a dictionary that contains the parameters for the dataset, which is optional and can be customized.
- The `dataloader_params` is a dictionary that contains the parameters for the dataloader, which is optional and can be customized.

    ```yaml
    dataset_config:
        train:
            type: MnistDataset  # Required
            dataset_params:  # Custom and Optional
            root: data
            split: true
            download: true
            dataloader_params: # Custom and Optional
            batch_size: 64
            shuffle: true
            num_workers: 4

        val:    # Optional
            type: MnistDataset  # Required
            dataset_params: # Custom and Optional
            root: data
            split: false
            download: true
            dataloader_params: # Custom and Optional
            batch_size: 100
            shuffle: false
            num_workers: 4
    ```

### train_model_config

- model_config including the model type and model parameters.
- The `type` is the model class name, which is required (should be add to the dictionary in `train_helper.py`).
- The `params` is a dictionary that contains the parameters for the model, which is optional and can be customized.

    ```yaml
    model_config:
        type: MnistModel # Required
        # params: # Custom and Optional
    ```

### train_config

- train_config including the training parameters.
- The `epochs` is the number of epochs to train the model, which is required.
- The `grad_clip` is the gradient clipping value, which is optional and can be null or not specified.
- The `optimizer` is a dictionary that contains the optimizer type and parameters, which is required (should be add to the dictionary in `train_helper.py`).
- The `scheduler` is a dictionary that contains the scheduler type and parameters, which is optional and can be null or not specified. If not specified, it will not use a scheduler (should be add to the dictionary in `train_helper.py`).
- The `early_stopping` is a dictionary that contains the early stopping parameters, which is optional and can be null or not specified. If not specified, it will not use early stopping (should be add to the dictionary in `train_helper.py`).

    ```yaml
    train_config:
        num_epochs: 15  # Required
        # grad_clip: 5.0 # Custom and Optional

        optimizer: # Required
            type: Adam # Required
            params: # Custom and Optional
            lr: 0.01
            weight_decay: 0.0001
            amsgrad: false

        scheduler: # Custom and Optional
            type: StepLR # Required
            params:
            step_size: 5
            gamma: 0.5

        early_stopping: # Custom and Optional
            patience: 3
            min_delta: 0.001
            mode: max
    ```

### train_evaluate_config

- evaluate_config including the evaluation metrics and main metric.
- The `metrics` is a list of dictionaries that contains the metric function name and parameters, which is at least one and can be customized (functions should be added to the dictionary in `evaluate_helper.py`).
- The `main_metric` is the main metric to use for evaluation, which is required in case of early stopping or saving the best model (the main_metric should be one that returns by the metrics functions). Metrics functions are defined in `src/evaluate/metrics.py` and should be return dict.

    ```yaml
    evaluate_config:
        metrics:
            - function: mnist_accuracy
            - function: mnist_f1_score
            params: # Custom and Optional
                average: weighted

        main_metric: accuracy # Required
    ```

## Test Configuration (**configs/test_config.yaml**)

### test_path_config

- the output directory and checkpoint path are required.
- The `output_dir` is the directory to save the test results, which is required.
- The `checkpoint_path` is the path to the checkpoint file, which is required to load the model for testing.

    ```yaml
    path_config:
        output_dir: outputs/test/mnist_experiment
        checkpoint_path: outputs/train/mnist_experiment/best_model.pth  # Required
    ```

### test_dataset_config

- dataset_config including the test dataset.
- The `test` dataset is required, and if not specified, it will raise an error.
- Each dataset has a type, dataset_params, and dataloader_params.
- The `type` is the dataset class name, which is required (should be add to the dictionary in `test_helper.py`).
- The `dataset_params` is a dictionary that contains the parameters for the dataset, which is optional and can be customized.
- The `dataloader_params` is a dictionary that contains the parameters for the dataloader, which is optional and can be customized.

    ```yaml
    dataset_config:
        test:
            type: MnistDataset
            dataset_params:
            root: data
            split: true
            download: true
            dataloader_params:
            batch_size: 64
            shuffle: false
            num_workers: 4
    ```

### test_model_config

- model_config including the model type and model parameters.
- The `type` is the model class name, which is required (should be add to the dictionary in `train_helper.py`).
- The `params` is a dictionary that contains the parameters for the model, which is optional and can be customized.

    ```yaml
    model_config:
        type: MnistModel
        # params: # Custom and Optional

    ```

### test_evaluate_config

- evaluate_config including the evaluation metrics and main metric.
- The `metrics` is a list of dictionaries that contains the metric function name and parameters, which is at least one and can be customized (functions should be added to the dictionary in `evaluate_helper.py`).
- The `main_metric` is the main metric to use for evaluation, which is required in case of saving the best model (the main_metric should be one that returns by the metrics functions). Metrics functions are defined in `src/evaluate/metrics.py` and should be return dict.

    ```yaml
    evaluate_config:
        metrics:
            - function: mnist_accuracy
            - function: mnist_f1_score
            params: # Custom and Optional
                average: macro

        main_metric: accuracy # Required
    ```

## Inference Configuration (**configs/inference_config.yaml**)

### inference_path_config

- path_config including the output directory and checkpoint path.
- The `output_dir` is the directory to save the inference results, which is required.
- The `checkpoint_path` is the path to the checkpoint file, which is required to load the model for inference.

    ```yaml
    path_config:
        output_dir: outputs/inference/mnist_experiment
        checkpoint_path: outputs/train/mnist_experiment/best_model.pth
    ```

### inference_dataset_config

- dataset_config including the inference dataset.
- The `inference` dataset is required, and if not specified, it will raise an error.
- Each dataset has a type, dataset_params, and dataloader_params.
- The `type` is the dataset class name, which is required (should be add to the dictionary in `inference_helper.py`).
- The `dataset_params` is a dictionary that contains the parameters for the dataset, which is optional and can be customized.
- The `dataloader_params` is a dictionary that contains the parameters for the dataloader, which is optional and can be customized.

    ```yaml
    dataset_config:
        inference:
            type: MnistDataset
            dataset_params:
            root: data
            split: false
            download: true
            dataloader_params:
            batch_size: 64
            shuffle: false
            num_workers: 4
    ```

### inference_model_config

- model_config including the model type and model parameters.
- The `type` is the model class name, which is required (should be add to the dictionary in `inference_helper.py`).
- The `params` is a dictionary that contains the parameters for the model, which is optional and can be customized.

    ```yaml
    model_config:
        type: MnistModel
        # params: # Custom and Optional
    ```

### inference_config

- inference_config including the inference mode, display, save, and log depend on your implementation.
- The `mode` is the inference mode, which is required and can be one of the following: `all`, `random`, `single`.
- The `display` is a boolean value that indicates whether to display the inference results, which is optional and can be true or false (should be true if only one image is displayed else false).
- The `save` is a boolean value that indicates whether to save the inference results, which is optional and can be true or false.
- The `log` is a boolean value that indicates whether to log the inference results, which is optional and can be true or false.
- if `mode` = `single` you can add `id` to the inference_config to specify the id of the image to be displayed.

    ```yaml
    inference_config:
        mode: "random"
        display: true
        save: false
        log: true
    ```

### inference_evaluate_config

- evaluate_config including the evaluation metrics and main metric.
- The `metrics` is a list of dictionaries that contains the metric function name and parameters, which is at least one and can be customized (functions should be added to the dictionary in `evaluate_helper.py`).
- The `evaluate_config` is optional and can be null or not specified. If not specified, it will not use evaluation (should be add to the dictionary in `inference_helper.py`). Should only be used if inference all.

    ```yaml
    evaluate_config:
        metrics:
            - function: mnist_accuracy
            - function: mnist_f1_score
            params: # Custom and Optional
                average: macro
    ```
