import os
import shutil
from pathlib import Path

# Đường dẫn chính xác đến các thư mục cần làm sạch (tương đối hoặc tuyệt đối)
FOLDERS_TO_CLEAN = [
    "logs",
    "outputs/test",
    "outputs/inference",
]

def clean_folders() -> None:
    deleted_dirs = 0

    for folder in FOLDERS_TO_CLEAN:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for item in folder_path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"🧹 Deleted folder: {item}")
                    deleted_dirs += 1
            except Exception as e:
                print(f"❌ Error deleting {item}: {e}")

    print(f"\n✅ Done. Deleted {deleted_dirs} folders inside specified folders.")

if __name__ == "__main__":
    clean_folders()
