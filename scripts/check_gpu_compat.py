#!/usr/bin/env python3
"""
Check GPU compatibility for FitnessRL training.
Run this on your VM to verify your GPU is compatible.
"""

import subprocess
import sys

def check_gpu_compatibility():
    print("=" * 70)
    print("GPU Compatibility Checker for FitnessRL Training")
    print("=" * 70)
    
    # Check if CUDA is available
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n✓ PyTorch installed: {torch.__version__}")
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            # Get GPU info
            gpu_count = torch.cuda.device_count()
            print(f"✓ Number of GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                vram_gb = gpu_props.total_memory / (1024**3)
                compute_cap = f"{gpu_props.major}.{gpu_props.minor}"
                
                print(f"\n{'='*70}")
                print(f"GPU {i}: {gpu_name}")
                print(f"{'='*70}")
                print(f"  VRAM: {vram_gb:.1f} GB")
                print(f"  Compute Capability: {compute_cap}")
                
                # Determine compatibility
                major, minor = gpu_props.major, gpu_props.minor
                
                if major == 10:  # Blackwell (B100, B200)
                    print(f"\n  ⚠️  BLACKWELL ARCHITECTURE DETECTED!")
                    print(f"  This GPU requires PyTorch nightly build.")
                    print(f"  See scripts/B200_SETUP.md for setup instructions.")
                    print(f"\n  Recommended action:")
                    print(f"    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
                    
                elif major >= 7:  # Compatible
                    print(f"\n  ✅ COMPATIBLE with standard PyTorch!")
                    
                    # Recommend model size based on VRAM
                    if vram_gb >= 80:
                        print(f"\n  💪 Recommended: 32B or 14B model")
                        print(f"     BASE_MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'")
                    elif vram_gb >= 40:
                        print(f"\n  💪 Recommended: 14B model")
                        print(f"     BASE_MODEL_NAME = 'Qwen/Qwen2.5-14B-Instruct'")
                    elif vram_gb >= 24:
                        print(f"\n  ✅ Recommended: 7B or 14B model (with quantization)")
                        print(f"     BASE_MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'")
                    elif vram_gb >= 16:
                        print(f"\n  ✅ Recommended: 7B model")
                        print(f"     BASE_MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'")
                    else:
                        print(f"\n  ⚠️  Warning: Low VRAM ({vram_gb:.1f}GB)")
                        print(f"     You may need quantization or a smaller model.")
                    
                    # GPU-specific recommendations
                    if "A100" in gpu_name:
                        print(f"\n  🌟 Excellent! A100 is perfect for training.")
                        print(f"     gpu_memory_utilization = 0.85")
                    elif "H100" in gpu_name:
                        print(f"\n  🌟 Outstanding! H100 is top-tier.")
                        print(f"     gpu_memory_utilization = 0.85")
                    elif "V100" in gpu_name:
                        print(f"\n  ✅ Good! V100 works well for 7B models.")
                        print(f"     gpu_memory_utilization = 0.75")
                    elif "T4" in gpu_name:
                        print(f"\n  ✅ T4 works for 7B models.")
                        print(f"     gpu_memory_utilization = 0.70")
                        print(f"     Consider using 4-bit quantization for better fit.")
                    elif "RTX" in gpu_name:
                        print(f"\n  ✅ RTX series works great!")
                        print(f"     gpu_memory_utilization = 0.80")
                    
                else:  # Older GPUs
                    print(f"\n  ⚠️  OLDER GPU (Compute {compute_cap})")
                    print(f"  May have limited support. Minimum is compute 7.0.")
                    
        else:
            print("\n❌ CUDA not available!")
            print("   Make sure you have:")
            print("   1. NVIDIA GPU installed")
            print("   2. NVIDIA drivers installed")
            print("   3. CUDA toolkit installed")
            print("   4. PyTorch with CUDA support")
            
    except ImportError:
        print("\n❌ PyTorch not installed!")
        print("   Install with: pip install torch")
        return
    
    # Check nvidia-smi
    print(f"\n{'='*70}")
    print("NVIDIA Driver Information")
    print("=" * 70)
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  nvidia-smi not available")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    if cuda_available and gpu_count > 0:
        major = torch.cuda.get_device_properties(0).major
        if major == 10:
            print("❌ ACTION REQUIRED: Install PyTorch nightly for Blackwell support")
            print("   See: scripts/B200_SETUP.md")
        elif major >= 7:
            print("✅ Your GPU is compatible! You can run training.")
            print("   Next steps:")
            print("   1. Ensure .env file has API keys")
            print("   2. Run: python scripts/rag_fitnessrl_art.py")
        else:
            print("⚠️  GPU may have limited compatibility")
    else:
        print("❌ No compatible GPU detected")
    
    print("=" * 70)


if __name__ == "__main__":
    check_gpu_compatibility()

