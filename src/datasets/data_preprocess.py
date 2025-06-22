import json
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_xml_to_json(data_dir, split="train", output_file="dataset.json"):
    if split == "train":
        annot_path = Path(data_dir) / "trainannot"
        data_path = Path(data_dir) / "trainimage"
    elif split == "test":
        annot_path = Path(data_dir) / "testannot"
        data_path = Path(data_dir) / "testimage"

    mapping_file = Path(data_dir) / f"mapping_{output_file}"
    output_file = Path(data_dir) / output_file

    dataset = []
    path_mapping = {}

    for xml_file in annot_path.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_path = Path(str(data_path / root.find("filename").text)).as_posix()

        bboxes = []
        labels_1 = []
        labels_2 = []

        for obj in root.findall("object"):
            bbox = [
                int(obj.find("bndbox/xmin").text),
                int(obj.find("bndbox/ymin").text),
                int(obj.find("bndbox/xmax").text),
                int(obj.find("bndbox/ymax").text)
            ]
            bboxes.append(bbox)
            labels_1.append(obj.find("name").text)
            labels_2.append(obj.find("name2").text)

        record = {
            'image_path': image_path,
            'boxes': bboxes,
            'labels_patho': labels_2,
            'labels_type': labels_1
        }
        dataset.append(record)

        path_mapping[image_path] = {
            'boxes': bboxes,
            'labels_patho': labels_2,
            'labels_type': labels_1
        }

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    with open(mapping_file, 'w') as f:
        json.dump(path_mapping, f, indent=4)

    print(f"✅ Saved {len(dataset)} samples to {output_file}")
    print(f"✅ Saved path mapping to {mapping_file}")

if __name__ == "__main__":
    convert_xml_to_json(data_dir=Path("data/full_dataset"), split="train", output_file="train_dataset.json")
    convert_xml_to_json(data_dir=Path("data/full_dataset"), split="test", output_file="test_dataset.json")
