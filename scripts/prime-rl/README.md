# Prime-RL Fitness Agent Training

Training scripts using TRL's GRPO and [PrimeIntellect's verifiers library](https://github.com/PrimeIntellect-ai/verifiers) for RL-based fitness agent training.

## Overview

Two scripts with different purposes:

| Script | Purpose | vLLM Required? |
|--------|---------|----------------|
| `train_grpo.py` | **Training** with GRPO + LoRA | ❌ No |
| `prime-rag-fitnessrl.py` | **Evaluation** with ToolEnv | ✅ Yes |

**Key point**: Training uses TRL's GRPOTrainer which handles generation internally. You don't need a separate vLLM server for training.

## Installation

```bash
# Core training dependencies
pip install trl peft transformers accelerate

# For W&B logging (recommended)
pip install wandb

# For evaluation script only (not needed for training)
pip install vllm verifiers

# Install project dependencies
uv sync
```

## Environment Setup

Create a `.env` file with:

```bash
# Required for both training and evaluation
PINECONE_API_KEY=your-pinecone-key

# Required for W&B logging
WANDB_API_KEY=your-wandb-key
```

### W&B Setup (Recommended)

```bash
wandb login
```

## Usage

### Training (No vLLM needed!)

Training uses TRL's GRPOTrainer with LoRA - everything runs on a single GPU:

```bash
# Memory optimization for large models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training (single GPU)
accelerate launch scripts/prime-rl/train_grpo.py

# Multi-GPU with DeepSpeed
accelerate launch --config_file configs/deepspeed_zero2.yaml scripts/prime-rl/train_grpo.py
```

**Memory requirements with LoRA:**
- Qwen2.5-14B: ~20-30GB GPU
- Qwen2.5-7B: ~12-16GB GPU

### Evaluation (Requires vLLM)

For running evaluation with the trained model:

```bash
# 1. Start vLLM server (requires ~40GB+ GPU memory for 14B)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8

# 2. Run evaluation in another terminal
python scripts/prime-rl/prime-rag-fitnessrl.py
```

**Note**: Training and evaluation should be run on **separate GPUs** if you want to run both simultaneously.

## Reward Functions

| Function | Weight | Description |
|----------|--------|-------------|
| `reward_schema` | 0.2 | Valid JSON schema with required fields |
| `reward_macros` | 0.5 | All macros within ±5% of targets |
| `reward_variety` | 0.15 | At least 3 unique meals |
| `reward_no_banned` | 0.15 | No banned ingredients in meal names |

## Tools

The agent has access to two tools:

1. **recipe_semantic_search(meal_query, k)**
   - Searches Pinecone for recipes matching the query
   - Returns JSON with name, calories, protein, carbs, fat

2. **return_final_answer(answer)**
   - Submits the final meal plan
   - Must be called at the end of the agent's reasoning

## Expected Output Format

```json
{
  "meals": [
    {
      "name": "Grilled Chicken with Rice and Broccoli",
      "quantity": 1.5,
      "calories": 780,
      "proteins": 67.5,
      "carbs": 82.5,
      "fats": 18,
      "sequence": 1
    },
    ...
  ]
}
```

## Configuration

### LoRA Configuration (Default)

The training script uses LoRA by default for memory-efficient training:

```python
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    r=16,                           # LoRA rank
    lora_alpha=32,                  # Scaling factor
    lora_dropout=0.05,              # Regularization
    target_modules=[                # All attention + MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
)
```

### Training Configuration

```python
# Model
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# W&B
WANDB_PROJECT = "fitness-agent-prime-rl"
WANDB_ENTITY = "your-team"  # or None for personal

# Training (memory-optimized defaults)
GRPOConfig(
    num_train_epochs=3,
    per_device_train_batch_size=1,     # Small batch, use grad accum
    gradient_accumulation_steps=16,     # Effective batch = 16
    learning_rate=1e-5,
    num_generations=2,                  # Rollouts per prompt
    max_completion_length=1024,         # Max tokens per response
    gradient_checkpointing=True,        # Save memory
    bf16=True,                          # Mixed precision
)
```

## W&B Dashboard

When training runs, W&B will log:

### Training Metrics
- `train/loss` - Policy loss
- `train/kl_divergence` - KL from reference policy
- `train/entropy` - Policy entropy
- `train/learning_rate` - Current LR

### Rollout Metrics
- `rollout/mean_reward` - Average reward per batch
- `rollout/schema_rate` - % with valid schema
- `rollout/macros_rate` - % meeting macro targets
- `rollout/variety_rate` - % with 3+ unique meals
- `rollout/no_banned_rate` - % avoiding banned ingredients
- `rollout/reward_histogram` - Reward distribution
- `rollout/turns_histogram` - Agent turns distribution

### Evaluation Metrics
- `eval/mean_reward` - Mean reward across eval set
- `eval/examples` - Table of example trajectories

### Artifacts
- Model checkpoints saved as artifacts
- Final model saved at end of training

## Comparison with ART

| Feature | ART (rag_fitnessrl_v2.py) | verifiers (prime-rl) |
|---------|---------------------------|----------------------|
| Tool calling | LangGraph ReAct agent | Native OpenAI tool calling |
| Reward | Custom async functions | Rubric with weighted funcs |
| Training | ART TrainableModel | GRPOTrainer |
| Inference | Built-in vLLM | External vLLM server |

## Troubleshooting

### Out of Memory (OOM) Errors

For large models like Qwen2.5-14B, use these memory-saving techniques:

1. **Reduce batch size and increase gradient accumulation:**
```python
GRPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
```

2. **Enable gradient checkpointing:**
```python
GRPOConfig(gradient_checkpointing=True)
```

3. **Use mixed precision:**
```python
GRPOConfig(bf16=True)  # or fp16=True for older GPUs
```

4. **Reduce generations and completion length:**
```python
GRPOConfig(
    num_generations=2,
    max_completion_length=1024,
)
```

5. **Use a smaller model for testing:**
```python
# Instead of 14B, try 7B or smaller
trainer = GRPOTrainer(model="Qwen/Qwen2.5-7B-Instruct", ...)
```

6. **Use LoRA/PEFT for parameter-efficient training:**
```python
from peft import LoraConfig
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
trainer = GRPOTrainer(model=MODEL_NAME, peft_config=peft_config, ...)
```

7. **Set environment variable for memory fragmentation:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### NCCL Issues
```bash
export NCCL_P2P_DISABLE=1
# or
export NCCL_CUMEM_ENABLE=1
```

### Too Many Open Files
```bash
ulimit -n 4096
```

### vLLM Memory Issues
Reduce `--gpu-memory-utilization` or use tensor parallelism.

## Loading Trained LoRA Adapter

After training, load your LoRA adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./outputs/prime-rl-fitness")

# Or merge into a single model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-fitness-model")

# Use with tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
```

## Citation

```bibtex
@misc{brown_verifiers_2025,
  author       = {William Brown},
  title        = {{Verifiers}: Environments for LLM Reinforcement Learning},
  howpublished = {\url{https://github.com/PrimeIntellect-ai/verifiers}},
  year         = {2025}
}
```

