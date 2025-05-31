import torch
import sys

print("=== CUDA Diagnostics ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available. Possible reasons:")
    print("1. PyTorch was installed without CUDA support")
    print("2. CUDA drivers are not properly installed")
    print("3. GPU is not CUDA-compatible")

# Test tensor creation
try:
    x = torch.randn(3, 3)
    print(f"\nCPU tensor created: {x.device}")
    
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        print(f"CUDA tensor created: {x_cuda.device}")
    else:
        print("Cannot create CUDA tensor - CUDA not available")
        
except Exception as e:
    print(f"Error creating tensors: {e}")