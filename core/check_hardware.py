import psutil
import platform
import shutil
import subprocess

import torch
from core.logger import Logger

def check_hardware(verbose=True):
    """
    Check if the hardware supports CUDA and if a GPU is available.
    Returns:
        - device: The device to be used for computation (CPU or GPU).
        - is_cuda: Boolean indicating if CUDA is available.
    """
    logger = Logger("check_hardware")
    if verbose:
        # CPU
        logger.info("================================================================")
        logger.info("🧠 CPU Info:")
        logger.info(f"  - Processor: {platform.processor()}")
        logger.info(f"  - Physical cores: {psutil.cpu_count(logical=False)}")
        logger.info(f"  - Logical cores: {psutil.cpu_count(logical=True)}")

        # RAM
        ram = psutil.virtual_memory()
        logger.info("💾 RAM Info:")
        logger.info(f"  - Total RAM: {ram.total / (1024**3):.2f} GB")

        # Disk
        total, used, free = shutil.disk_usage("/")
        logger.info("💽 Disk Info:")
        logger.info(f"  - Total Disk: {total / (1024**3):.2f} GB")
        logger.info(f"  - Free Space: {free / (1024**3):.2f} GB")

        # GPU - PyTorch
        logger.info("🖥️ GPU (via PyTorch):")
        if torch.cuda.is_available():
            logger.info(f"  - CUDA available: ✅ Yes")
            logger.info(f"  - GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("  - CUDA available: ❌ No")

        # CUDA Toolkit
        logger.info("⚙️ CUDA Toolkit:")
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(result.stdout.strip())
            else:
                logger.info("  - nvcc not found (CUDA Toolkit may not be installed)")
        except FileNotFoundError:
            logger.info("  - nvcc not found (CUDA Toolkit may not be installed)")

        # cuDNN
        logger.info("📦 cuDNN Info:")
        try:
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"  - cuDNN version: {cudnn_version}")
        except Exception:
            logger.info("  - cuDNN not found or not available")

        # Python & Library versions
        logger.info("📦 Python & Library Versions:")
        logger.info(f"  - Python: {platform.python_version()}")
        logger.info(f"  - PyTorch: {torch.__version__}")
        logger.info(f"  - CUDA: {torch.version.cuda}")
        logger.info(f"  - cuDNN: {torch.backends.cudnn.version()}")

        logger.info("✅ Done")
        logger.info("================================================================")


    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    return device, is_cuda

if __name__ == "__main__":
    check_hardware()