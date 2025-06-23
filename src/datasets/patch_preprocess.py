import os
import json
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

from core.utils import set_seed

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def crop_patch(image, center, patch_size):
    half = patch_size // 2
    x, y = center
    left = max(0, x - half)
    upper = max(0, y - half)
    right = min(image.width, x + half)
    lower = min(image.height, y + half)
    return image.crop((left, upper, right, lower)), (left, upper, right, lower)

def sample_positive_patches(img, boxes, labels_patho, labels_type, patch_size,
                         nb_ps, output_size=None, pos_cutoff=0.75,
                         max_trials=10000):
    patches = []
    for idx, (box, patho_label, type_label) in enumerate(zip(boxes, labels_patho, labels_type)):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        sampled = 0
        trials = 0

        while sampled < nb_ps:
            x = random.randint(x1, x1+w)
            y = random.randint(y1, y1+h)
            trials += 1

            if trials > max_trials:
                print("Nb of trials reached maximum, decrease overlap cutoff by 0.05")
                pos_cutoff -= .05
                trials = 0
                if pos_cutoff <= .0:
                    print(sampled, "ROI patches sampled")
                    break

            x_1_patch = x - patch_size // 2
            y_1_patch = y - patch_size // 2
            x_2_patch = x + patch_size // 2
            y_2_patch = y + patch_size // 2
            patch_box = [x_1_patch, y_1_patch, x_2_patch, y_2_patch]

            overlap = compute_iou(patch_box, box)
            # print(patch_box, box, overlap)
            # exit(0)

            if overlap >= pos_cutoff:
                patch, coords = crop_patch(img, (x, y), patch_size)
                if output_size:
                    patch = patch.resize(output_size, Image.LANCZOS)
                patches.append({
                    'patch': patch,
                    'coords': coords,
                    'label_patho': patho_label,
                    'label_type': type_label,
                    'is_roi': True
                })
                sampled += 1

    return patches


def sample_hard_negatives(img, boxes, labels_patho, labels_type, patch_size,
                          nb_bkg=100, neg_cutoff=0.35, output_size=None,
                          max_trials=10000):
    patches = []
    for idx, (box, patho_label, type_label) in enumerate(zip(boxes, labels_patho, labels_type)):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        # Expand search area around the box
        search_x1 = max(0, x1 - patch_size)
        search_y1 = max(0, y1 - patch_size)
        search_x2 = min(img.size[1], x2 + patch_size)
        search_y2 = min(img.size[0], y2 + patch_size)

        sampled = 0
        trials = 0

        while sampled < nb_bkg:
            x = random.randint(search_x1, search_x2)
            y = random.randint(search_y1, search_y2)
            trials += 1

            if trials > max_trials:
                print(f"Max trials reached for box {idx}, found {sampled} negatives")
                break

            # Calculate patch coordinates
            x1_patch = x - patch_size // 2
            y1_patch = y - patch_size // 2
            x2_patch = x + patch_size // 2
            y2_patch = y + patch_size // 2

            # Skip if patch goes out of image bounds
            if (x1_patch < 0 or y1_patch < 0 or
                x2_patch >= img.size[1] or y2_patch >= img.size[0]):
                continue

            patch_box = [x1_patch, y1_patch, x2_patch, y2_patch]
            overlap = compute_iou(patch_box, box)

            if overlap <= neg_cutoff:
                patch_img, coords = crop_patch(img, (x, y), patch_size)

                if output_size:
                    patch_img = patch_img.resize(output_size, Image.LANCZOS)

                patches.append({
                    'patch': patch_img,
                    'coords': coords,
                    'label_patho': patho_label,
                    'label_type': type_label,
                    'is_roi': False  # Mark as negative sample
                })
                sampled += 1

    return patches

def sample_blob_negatives(img, boxes, patch_size,
                          nb_blob=100, neg_cutoff=0.35, output_size=None):
    img_np = np.array(img)
    height, width = img_np.shape[0], img_np.shape[1]

    # Setup blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 500
    params.filterByCircularity = False
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    if len(img_np.shape) == 3:  # Color image
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:  # Grayscale
        gray_img = img_np

    # Normalize and convert to 8-bit
    gray_img = (gray_img / gray_img.max() * 255).astype('uint8')
    keypoints = detector.detect(gray_img)

    # Convert keypoints to list and shuffle
    keypoints_list = list(keypoints)  # Chuyển tuple sang list
    random.shuffle(keypoints_list)    # Giờ có thể shuffle được

    patches = []
    sampled = 0

    for kp in keypoints_list:
        if sampled >= nb_blob:
            break

        x, y = int(kp.pt[0]), int(kp.pt[1])
        x1_patch = x - patch_size // 2
        y1_patch = y - patch_size // 2
        x2_patch = x + patch_size // 2
        y2_patch = y + patch_size // 2

        # Skip if patch goes out of image bounds
        if (x1_patch < 0 or y1_patch < 0 or
            x2_patch >= width or y2_patch >= height):
            continue

        # Check overlap with all ROIs
        is_valid = True
        patch_box = [x1_patch, y1_patch, x2_patch, y2_patch]
        for box in boxes:
            overlap = compute_iou(patch_box, box)
            if overlap > neg_cutoff:
                is_valid = False
                break

        if is_valid:
            # Crop using PIL (better for image quality)
            patch_img = img.crop((x1_patch, y1_patch, x2_patch, y2_patch))

            if output_size:
                patch_img = patch_img.resize(output_size, Image.LANCZOS)

            patches.append({
                'patch': patch_img,
                'coords': (x, y),
                'label_patho': None,  # Negative class
                'label_type': None,   # Negative class
                'is_roi': False    # Mark as negative sample
            })
            sampled += 1

    return patches

def save_patches(patches, output_dir, img_name):
    """Lưu patches và trả về metadata"""
    metadata = []

    for i, patch_data in enumerate(patches):
        patch_filename = f"{img_name}_{i:04d}.png"
        patch_path = Path(output_dir)/patch_filename
        patch_data['patch'].save(patch_path)

        metadata.append({
            "patch_path": str(patch_path),
            "original_image": img_name,
            "bbox": patch_data['coords'],
            "label_patho": patch_data['label_patho'],
            "label_type": patch_data['label_type'],
            "is_roi": patch_data.get('is_roi', False),
            "is_hard_negative": patch_data.get('is_hard_negative', False)
        })

    return metadata

def process_image(obj, output_dir, patch_size, nb_ps, nb_blob, nb_hns,
                 pos_cutoff, neg_cutoff, output_size):
    """Xử lý một ảnh và tạo các patches"""
    img_path = obj['image_path']
    boxes = obj.get('boxes', [])
    labels_patho = obj.get('labels_patho', [])
    labels_type = obj.get('labels_type', [])

    img_name = Path(img_path).stem

    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Không mở được ảnh {img_path}: {e}")
        return []

    # Tạo ROI patches
    roi_patches = sample_positive_patches(
        img, boxes, labels_patho, labels_type,
        patch_size, nb_ps, output_size, pos_cutoff=pos_cutoff
    )

    # Tạo hard negative patches
    bkg_patches = sample_hard_negatives(
        img, boxes, labels_patho, labels_type,
        patch_size, nb_hns, neg_cutoff, output_size
    )

    # Tạo blob patches
    # hn_patches = sample_blob_negatives(
    #     img, boxes, patch_size, nb_blob, neg_cutoff=neg_cutoff, output_size=None
    # )


    # Kết hợp tất cả patches
    all_patches = roi_patches + bkg_patches + hn_patches

    # Lưu patches và lấy metadata
    return save_patches(all_patches, output_dir, img_name)

def crop_patches(
    json_path,
    output_dir="patches",
    meta_json_path="patches_metadata.json",
    patch_size=256,
    nb_ps=30,
    nb_blob=30,
    nb_hns=15,
    pos_cutoff=0.75,
    neg_cutoff=0.35,
    output_size=None
):
    """Main function để xử lý toàn bộ dataset"""
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    all_metadata = []

    for obj in tqdm(data, desc="Processing images"):
        all_metadata.extend(
            process_image(
                obj, output_dir, patch_size, nb_ps, nb_blob, nb_hns,
                pos_cutoff, neg_cutoff, output_size
            )
        )

    # Lưu metadata
    with open(meta_json_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Đã tạo {len(all_metadata)} patches:")
    print(f"- ROI patches: {len([m for m in all_metadata if m['is_roi']])}")
    print(f"- Hard negatives: {len([m for m in all_metadata if not m['is_roi'] and not m['is_hard_negative']])}")
    print(f"- Blob negatives: {len([m for m in all_metadata if m['is_hard_negative']])}")


if __name__ == "__main__":
    set_seed(42)

    crop_patches(
        json_path='data/full_dataset/train_dataset.json',
        output_dir='data/patch_dataset/patches',
        meta_json_path='data/patch_dataset/patches_metadata.json',
        patch_size=128,
        nb_ps=30,
        nb_blob=30,
        nb_hns=15,
        pos_cutoff=0.75,
        neg_cutoff=0.35,
        output_size=(224, 224)
    )
