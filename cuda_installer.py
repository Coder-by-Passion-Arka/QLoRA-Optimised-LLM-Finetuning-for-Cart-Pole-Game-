import torch
import subprocess
import sys

def install_cuda_torch():
    print("CUDA is not available. Attempting to install the appropriate torch version with CUDA support...")
    try:
        # Use matching CUDA versions (cu118) for all packages
        subprocess.check_call([
            "uv", "pip", "install",
            "torch==2.2.2+cu118",
            "torchvision==0.17.2+cu118",
            "torchaudio==2.2.2+cu118",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        print("PyTorch with CUDA support installed. Please restart your Python session.")
    except Exception as e:
        print(f"Failed to install CUDA-enabled torch: {e}")
        print("\nTrying alternative installation method...")
        try:
            # Alternative installation using regular pip
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch==2.2.2+cu118",
                "torchvision==0.17.2+cu118",
                "torchaudio==2.2.2+cu118",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("PyTorch with CUDA support installed using pip. Please restart your Python session.")
        except Exception as e2:
            print(f"Both installation methods failed. Error: {e2}")

if not torch.cuda.is_available():
    install_cuda_torch()
else:
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:      {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    try:
        subprocess.check_call("nvidia-smi", shell=True)
    except:
        print("nvidia-smi command failed. Please ensure NVIDIA drivers are installed.")