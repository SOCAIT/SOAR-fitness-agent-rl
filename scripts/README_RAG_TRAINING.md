# RAG FitnessRL Training Guide

This guide explains how to run the RAG-based FitnessRL training script on a remote VM.

## Prerequisites

### 1. System Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM (e.g., V100, A100, or T4)
- **RAM**: At least 32GB
- **Storage**: At least 50GB free space
- **OS**: Linux (Ubuntu 20.04+ recommended)

### 2. Environment Setup

```bash
# Clone the repository
cd /path/to/your/workspace
git clone <your-repo-url>
cd fitness-reasoning-rl-agent

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Optional (for W&B logging)
WANDB_API_KEY=...
```

### 4. Data Setup

Ensure `data/fitness_scenarios.jsonl` exists in your project directory. This file contains the training scenarios.

## Running the Training Script

### Basic Usage

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the training script
python scripts/rag_fitnessrl_art.py
```

### Using Screen/Tmux (Recommended for Remote Sessions)

Since training can take hours, use `screen` or `tmux` to keep the process running:

```bash
# Using screen
screen -S fitness-training
source .venv/bin/activate
python scripts/rag_fitnessrl_art.py

# Detach: Ctrl+A, then D
# Reattach: screen -r fitness-training

# OR using tmux
tmux new -s fitness-training
source .venv/bin/activate
python scripts/rag_fitnessrl_art.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t fitness-training
```

## Monitoring

### Checking GPU Usage

```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Or use gpustat (if installed)
gpustat -i 1
```

### Viewing Logs

The script outputs progress directly to stdout. If you want to save logs:

```bash
python scripts/rag_fitnessrl_art.py 2>&1 | tee training.log
```

### Weights & Biases (W&B)

If you set `WANDB_API_KEY`, metrics will be logged to W&B automatically. View them at:
https://wandb.ai/your-username/fitness-agent-langgraph-rag

## Training Configuration

You can modify these parameters in the script's `main()` function:

```python
training_config = {
    "groups_per_step": 4,         # Number of scenario groups per training step
    "num_epochs": 3,              # Number of epochs to train
    "rollouts_per_group": 8,      # Number of rollouts per scenario group
    "learning_rate": 1e-5,        # Learning rate
    "max_steps": 150,             # Maximum training steps
}
```

## Output

### Model Checkpoints

Checkpoints are saved in `.art/` directory:
- `.art/fitness-agent-langgraph-7B-qwen2.5-004/`

### Training Artifacts

- Model checkpoints (every few steps)
- Training metrics (if W&B is enabled)
- Log files (if redirected to file)

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `gpu_memory_utilization` in the script (currently 0.8)
2. Reduce `rollouts_per_group` (currently 8)
3. Reduce `groups_per_step` (currently 4)

### CUDA Errors

```bash
# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
nvcc --version
```

### Pinecone Connection Issues

Ensure your Pinecone API key is correct and the indexes exist:
- `syntrafit-recipes-nutrition`
- `syntrafit-exercises`

### Import Errors

```bash
# Reinstall dependencies
uv sync --reinstall
```

## Stopping Training

- **If running in foreground**: `Ctrl+C`
- **If running in screen/tmux**: Attach to the session and press `Ctrl+C`

## Resuming Training

The script will automatically resume from the last checkpoint if you run it again with the same model name.

## Performance Tips

1. **Use a persistent disk** to save checkpoints
2. **Enable W&B logging** for better tracking
3. **Monitor GPU usage** to optimize memory utilization
4. **Use preemptible/spot instances** for cost savings (with auto-resume)

## Example: Running on Google Cloud VM

```bash
# SSH into VM
gcloud compute ssh your-vm-name --zone=us-central1-a

# Setup
cd /home/yourusername/fitness-reasoning-rl-agent
source .venv/bin/activate

# Run in screen
screen -S training
python scripts/rag_fitnessrl_art.py

# Detach and logout
# Ctrl+A, then D
exit
```

## Support

For issues or questions:
- Check logs for error messages
- Verify environment variables are set correctly
- Ensure GPU drivers and CUDA are properly installed
- Check the main README.md for general setup issues

