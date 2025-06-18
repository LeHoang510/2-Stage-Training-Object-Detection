# Structure of the Project

## Project layout

```python
# Project structure
C:.
├───.editorconfig
├───.gitignore
├───.gitleaks.toml
├───pyproject.toml
├───README.md
├───setup.py
├───tree.txt
│
├───core
│   ├───__init__.py
│   ├───logger.py
│   ├───utils.py
│   └───check_hardware.py
├───data
│   └───MNIST (your_dataset)
│
├───docs
│   ├───config.md
│   ├───guide.md
│   └───structure.md
│
├───logs
│   ├───inference
│   ├───train
│   └───test
│
├───models
│   ├───__init__.py
│   └───mnist (your_model)
│       ├───mnist_model.py
│       └───__init__.py
│
├───notebooks
├───outputs
│   ├───inference
│   ├───test
│   └───train
│
├───requirements
│   ├───requirements-build.txt
│   ├───requirements-dev.txt
│   ├───requirements-docs.txt
│   ├───requirements-test.txt
│   └───requirements.txt
│
├───scripts
│       clean.py
│
└───src
    ├───configs
    │   └───mnist (your_configs)
    │       ├───mnist_inference_config.yaml
    │       ├───mnist_test_config.yaml
    │       └───mnist_train_config.yaml
    │
    ├───datasets
    │   ├───__init__.py
    │   ├───template_preprocess.py
    │   └───mnist_dataset.py
    │
    ├───evaluate
    │   └───mnist_evaluate.py
    ├───inference
    │   └───mnist_inference.py
    ├───test
    │   └───mnist_test.py
    ├───train
    │   └───mnist_train.py
    ├───visualize
    │   └───visualize.py
    └───utils
        ├───config_validator.py
        ├───evaluate_helper.py
        ├───train_helper.py
        ├───visualize_helper.py
        └───__init__.py

```

## Directory and File Descriptions

### Project root directory

- **core/**: Contains core utilities such as logging, hardware checks, and utility functions.
- **data/**: Contains datasets used for training and evaluation.
- **docs/**: Contains documentation files.
- **logs/**: Contains log files for different stages of the project.
- **models/**: Contains model definitions and related files.
- **notebooks/**: Contains Jupyter notebooks for exploration and experimentation.
- **outputs/**: Contains output files from various stages of the project.
- **requirements/**: Contains requirement files for different environments.
- **scripts/**: Contains utility scripts for various tasks.
- **src/**: Contains the main source code for the project.

- **setup.py**: The setup script for the project, used for installation and packaging.
- **pyproject.toml**: The project configuration file, used for managing dependencies and build settings.
- **.editorconfig**: Configuration file for code style and formatting.
- **.gitignore**: Specifies files and directories that should be ignored by Git.
- **.gitleaks.toml**: Configuration file for gitleaks, used to detect sensitive information in the codebase.

### src directory

- **configs/**: Contains configuration files for different tasks (training, testing, inference).
- **datasets/**: Contains dataset-related files, including dataset loading and preprocessing scripts.
- **evaluate/**: Contains evaluation scripts for assessing model performance.
- **inference/**: Contains scripts for running inference on trained models.
- **test/**: Contains test scripts for evaluating model performance on test datasets.
- **train/**: Contains training scripts for training models.
- **visualize/**: Contains scripts for visualizing results and model performance.
- **utils/**: Contains utility functions and helpers for various tasks, such as configuration validation, evaluation, training, and visualization.
