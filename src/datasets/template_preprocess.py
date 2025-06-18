import os
from pathlib import Path

from core.logger import Logger
from core.utils import load_json, save_json

def preprocess_template_raw_data(raw_data_path: str, output_path: str):
    """
    Preprocess the raw data for the template dataset.
    This function should be implemented to handle the specific preprocessing steps required for the dataset.
    """
    # Iterate through the raw data files
    raw_data_path = Path(raw_data_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    for file in raw_data_path.glob("*.json"):
        # Load the raw data
        raw_data = load_json(file)

        # Perform preprocessing steps (this is a placeholder, implement as needed)
        processed_data = {
            "image_id": raw_data.get("id"),
            "image_path": raw_data.get("image_path").strip(),
            "label": raw_data.get("label")
        }

        # Save the processed data
        output_file = output_path / file.name
        save_json(processed_data, output_file)
        logger.info(f"Processed {file.name} and saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    logger = Logger("preprocess")
    raw_data_path = "path/to/raw/data"
    output_path = "path/to/output/data"
    preprocess_template_raw_data(raw_data_path, output_path)
    logger.info("Preprocessing completed")
