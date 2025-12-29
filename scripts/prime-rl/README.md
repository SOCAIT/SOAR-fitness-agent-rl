# Prime-RL Fitness Agent Training

Training scripts using [PrimeIntellect's verifiers library](https://github.com/PrimeIntellect-ai/verifiers) for RL-based fitness agent training.

## Overview

The `verifiers` library provides a clean abstraction for:
- **ToolEnv**: Agentic tool-calling environments
- **Rubric**: Composable reward functions with weights
- **GRPOTrainer**: Group Relative Policy Optimization training

## Files

- `prime-rag-fitnessrl.py` - Main evaluation script with ToolEnv
- `train_grpo.py` - Full GRPO training script

## Installation

```bash
# Install verifiers (for environments and rubrics)
pip install verifiers

# Install trl for GRPO training
pip install trl

# Install vLLM for inference
pip install vllm

# Install W&B for logging
pip install wandb

# Install project dependencies
uv sync
```

## Environment Setup

Create a `.env` file with:

```bash
# Required
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-openai-key  # or dummy for local vLLM

# Optional - W&B logging
WANDB_API_KEY=your-wandb-key
WANDB_ENTITY=your-username-or-team  # optional

# Optional - vLLM
VLLM_BASE_URL=http://localhost:8000/v1
```

### W&B Setup (Recommended)

1. Create account at [wandb.ai](https://wandb.ai)
2. Get API key from [wandb.ai/authorize](https://wandb.ai/authorize)
3. Add to `.env` or run:
   ```bash
   wandb login
   ```

## Usage

### 1. Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8
```

### 2. Run Evaluation

```bash
python scripts/prime-rl/prime-rag-fitnessrl.py
```

### 3. Run Full Training

```bash
# Single GPU
accelerate launch scripts/prime-rl/train_grpo.py

# Multi-GPU with DeepSpeed
accelerate launch --config_file configs/deepspeed_zero2.yaml scripts/prime-rl/train_grpo.py
```

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

Edit the configuration variables at the top of each script:

```python
# Model
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
VLLM_BASE_URL = "http://localhost:8000/v1"

# W&B
WANDB_PROJECT = "fitness-agent-prime-rl"
WANDB_ENTITY = "your-team"  # or None for personal

# Training
GRPO_CONFIG = GRPOConfig(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_generations=4,  # Rollouts per prompt
    max_new_tokens=2048,
    temperature=0.7,
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

## Citation

```bibtex
@misc{brown_verifiers_2025,
  author       = {William Brown},
  title        = {{Verifiers}: Environments for LLM Reinforcement Learning},
  howpublished = {\url{https://github.com/PrimeIntellect-ai/verifiers}},
  year         = {2025}
}
```

