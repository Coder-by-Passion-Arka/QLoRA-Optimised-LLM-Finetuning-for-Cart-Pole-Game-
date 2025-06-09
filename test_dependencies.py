import torch
import triton
import bitsandbytes as bnb
from unsloth import FastLanguageModel

def test_imports():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Triton version: {triton.__version__}")
    print(f"BitsAndBytes version: {bnb.__version__}")
    
if __name__ == "__main__":
    test_imports()