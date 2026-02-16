import torch
import sys
import os
import time

print("\n=== DIAGNOSTIC START ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor creation
    try:
        t = torch.tensor([1.0], device="cuda")
        print("Tensor on CUDA: Success")
        
        # Test performance
        print("Running simple performance test...")
        start = time.time()
        for _ in range(10000):
            t = t + 1
        torch.cuda.synchronize()
        end = time.time()
        print(f"Simple CUDA Loop (10k ops) Time: {end - start:.4f}s")
        if (end - start) > 1.0:
            print("WARNING: CUDA seems very slow!")
        else:
            print("Performance looks normal (fast).")
            
    except Exception as e:
        print(f"Tensor on CUDA: Failed ({e})")
else:
    print("CUDA NOT AVAILABLE - falling back to CPU")

print(f"Environment: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print("=== DIAGNOSTIC END ===\n")
