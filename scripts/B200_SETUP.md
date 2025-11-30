# NVIDIA B200 GPU Setup Guide

You have an **NVIDIA B200** with compute capability 10.0 - one of the most powerful GPUs available! 🚀

## The Problem

The B200 (Blackwell architecture) is so new that standard PyTorch installations don't have CUDA kernels compiled for compute capability 10.0 (sm_100). This causes the error:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

## Solutions

### ✅ Option 1: Install PyTorch Nightly (RECOMMENDED)

```bash
cd /root/SOAR-fitness-agent-rl
source .venv/bin/activate

# Remove old PyTorch
pip uninstall torch torchvision torchaudio vllm -y

# Install PyTorch nightly with CUDA 12.4+ (has sm_100 support)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Reinstall vLLM to build against new PyTorch
pip install vllm --no-build-isolation

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### ✅ Option 2: Use NVIDIA Container (EASIEST)

NVIDIA's official containers have full B200 support:

```bash
# Pull latest NVIDIA PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.11-py3

# Run your training
nvidia-docker run --gpus all -it --rm \
  -v /root/SOAR-fitness-agent-rl:/workspace \
  -w /workspace \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e PINECONE_API_KEY=$PINECONE_API_KEY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  nvcr.io/nvidia/pytorch:24.11-py3 \
  bash -c "pip install uv && uv sync && python scripts/rag_fitnessrl_art.py"
```

### ✅ Option 3: Use Environment Variables

The script now automatically sets these, but you can also set them manually:

```bash
export TORCH_CUDA_ARCH_LIST="9.0;10.0"
export VLLM_ALLOW_RUNTIME_COMPILATION="1"
export CUDA_LAUNCH_BLOCKING="1"  # For debugging only

python scripts/rag_fitnessrl_art.py
```

## Optimizing for B200

Your B200 likely has **192GB HBM3e memory** - here's how to take full advantage:

### 1. Use Larger Models

```python
# In rag_fitnessrl_art.py, change to:
BASE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"  # or even 32B!
MODEL_NAME = "fitness-agent-langgraph-14B-qwen2.5-000"
```

### 2. Increase Batch Size

```python
training_config = {
    "groups_per_step": 8,         # Up from 4
    "rollouts_per_group": 16,     # Up from 8
    # ... other settings
}
```

### 3. Use More GPU Memory

Already set to 0.85 in the updated script:

```python
gpu_memory_utilization=0.85,  # B200 has 192GB VRAM!
```

### 4. Enable Tensor Parallelism (for multi-GPU)

If you have multiple B200s:

```python
engine_args=art.dev.EngineArgs(
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    tensor_parallel_size=2,  # for 2 GPUs
),
```

## Verification Steps

After installing PyTorch nightly, verify everything works:

```bash
# Check PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"

# Check CUDA version
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# Test a simple CUDA operation
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print('CUDA works:', (x @ x).shape)"
```

## B200 Specs (for reference)

- **Architecture**: Blackwell (compute capability 10.0)
- **Memory**: 192GB HBM3e
- **Memory Bandwidth**: 8 TB/s
- **FP8 Performance**: ~9 petaFLOPS
- **Perfect for**: Large model training, long context lengths

## Recommended Configuration

For optimal performance on B200:

```python
# In the script
BASE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "fitness-agent-langgraph-14B-qwen2.5-000"

training_config = {
    "groups_per_step": 8,         # More parallel groups
    "num_epochs": 3,
    "rollouts_per_group": 12,     # More rollouts
    "learning_rate": 1e-5,
    "max_steps": 150,
}

model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=16384,     # Longer context! B200 can handle it
    ),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    ),
)
```

## Troubleshooting

### Still Getting CUDA Errors?

1. **Check PyTorch version**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   Should be 2.5.0+cu124 or newer

2. **Clear CUDA cache**:
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Reinstall with forced rebuild**:
   ```bash
   pip install --force-reinstall --no-cache-dir vllm
   ```

### OOM Errors?

Even with 192GB, you might hit OOM with very large batches:

- Reduce `gpu_memory_utilization` to 0.75
- Reduce `rollouts_per_group`
- Enable gradient checkpointing (if supported)

## Next Steps

1. ✅ Install PyTorch nightly (Option 1 above)
2. ✅ Pull latest code: `git pull`
3. ✅ Test with 7B model first
4. ✅ Scale up to 14B or 32B model
5. ✅ Increase batch sizes to maximize GPU utilization

Your B200 is incredibly powerful - make the most of it! 🚀

