# 🔧 Troubleshooting Guide

This guide helps you resolve common issues when using the Fitness Reasoning RL Agent.

## 🚀 Installation Issues

### Problem: `uv: command not found`

**Solution:**
```bash
# Install uv manually
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
uv --version
```

### Problem: Python version mismatch

**Error:** `requires-python >=3.10`

**Solution:**
```bash
# Check your Python version
python --version

# If using multiple Python versions:
python3.10 -m venv .venv
source .venv/bin/activate
```

### Problem: `pip install` fails with dependency conflicts

**Solution:**
```bash
# Use uv instead (handles conflicts better)
uv pip install -e .

# Or upgrade pip and try again
pip install --upgrade pip
pip install -e .
```

## 🔑 API Key Issues

### Problem: `Missing required environment variables: OPENAI_API_KEY, PINECONE_API_KEY`

**Solution:**

1. **Create .env file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys:**
   ```bash
   # Edit .env with your actual keys
   export OPENAI_API_KEY=sk-your_key_here
   export PINECONE_API_KEY=your_key_here
   ```

3. **Load environment variables:**
   ```bash
   source .env
   python train.py
   ```

### Problem: API keys are being exposed in git

**Solution:**
```bash
# Ensure .env is in .gitignore
echo ".env" >> .gitignore

# Remove accidentally committed .env
git rm --cached .env
git commit -m "Remove .env from tracking"
```

### Problem: "Invalid API key" when connecting to OpenAI/Pinecone

**Solution:**
- Verify your keys are correct at:
  - OpenAI: https://platform.openai.com/api-keys
  - Pinecone: https://app.pinecone.io/
- Ensure keys have required permissions
- Check API key hasn't expired or been revoked

## 🤖 Training Issues

### Problem: `CUDA out of memory`

**Solution:**

1. **Reduce batch size:**
   ```python
   # Edit src/config.py
   @dataclass
   class TrainingConfig:
       groups_per_step: int = 1  # Reduce from 2
       rollouts_per_group: int = 2  # Reduce from 4
   ```

2. **Reduce sequence length:**
   ```python
   @dataclass
   class ModelConfig:
       max_seq_length: int = 4096  # Reduce from 8192
   ```

3. **Enable memory optimization:**
   ```python
   gpu_memory_utilization: float = 0.6  # Reduce from 0.8
   ```

4. **Use CPU for testing:**
   ```python
   enforce_eager: bool = False
   ```

### Problem: Training is very slow

**Solution:**

1. **Check GPU utilization:**
   ```bash
   nvidia-smi -l 1  # Monitor every second
   ```

2. **If GPU not being used, verify CUDA:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Reduce model size for testing:**
   ```python
   base_model: str = "Qwen/Qwen2.5-3B-Instruct"  # Use smaller model
   ```

4. **Verify data loading isn't bottleneck:**
   ```bash
   # Add profiling in train.py
   import cProfile
   cProfile.run('train_agent(...)')
   ```

### Problem: `No module named 'art'`

**Solution:**
```bash
# Reinstall art with backend and langgraph support
pip install "openpipe-art[backend,langgraph]>=0.4.11"

# Verify installation
python -c "import art; print(art.__version__)"
```

### Problem: `FileNotFoundError: fitness_scenarios.jsonl not found`

**Solution:**
```bash
# Create symbolic link if using different data file
ln -s /path/to/your/scenarios.jsonl fitness_scenarios.jsonl

# Or provide via train_rl.sh
bash train_rl.sh --data-file /path/to/scenarios.jsonl
```

## 🗄️ Pinecone/Vector DB Issues

### Problem: `pinecone.exceptions.ApiException: 401 Unauthorized`

**Solution:**
1. Verify Pinecone API key is correct
2. Check API key has access to the specified index
3. Verify index names match your Pinecone setup:
   ```python
   # Edit src/config.py
   recipe_index_name: str = "your-index-name"
   exercise_index_name: str = "your-exercise-index"
   ```

### Problem: `Index not found`

**Solution:**
```bash
# Create index in Pinecone dashboard or via CLI
# Visit: https://app.pinecone.io/

# Or create via Python:
import pinecone
pinecone.create_index("syntrafit-recipes", dimension=1536)
```

### Problem: Recipe search returns no results

**Solution:**
1. Verify Pinecone index is populated with data
2. Check dimension matches embedding model (typically 1536 for OpenAI)
3. Try simple query first:
   ```python
   # In src/agent/tools.py
   results = pinecone_service.search("chicken rice", top_k=1)
   ```

## 📊 Weights & Biases Issues

### Problem: `wandb: disabled` warning during training

**Solution:**
```bash
# Either set WANDB_API_KEY
export WANDB_API_KEY=your_key

# Or disable warning (optional)
wandb disabled
```

### Problem: Metrics not showing in W&B dashboard

**Solution:**
1. Verify W&B account is active
2. Check WANDB_API_KEY is valid
3. Verify weave initialization:
   ```python
   weave.init(model.project, settings={"print_call_link": False})
   ```

## 🐛 Agent Execution Issues

### Problem: Agent times out or runs forever

**Solution:**

1. **Reduce max turns:**
   ```python
   @dataclass
   class AgentConfig:
       max_turns: int = 10  # Reduce from 30
   ```

2. **Add timeout:**
   ```python
   # In agent execution
   import asyncio
   await asyncio.wait_for(agent.invoke(...), timeout=60)
   ```

3. **Check LangGraph agent creation:**
   ```python
   from langgraph.prebuilt import create_react_agent
   react_agent = create_react_agent(chat_model, tools)
   ```

### Problem: Agent returns invalid JSON

**Solution:**
1. Check schema validation in rewards:
   ```python
   schema_score, info = verify_meal_plan_schema(payload)
   ```

2. Force valid JSON in prompt:
   ```python
   # Edit src/config.py get_planner_prompt()
   "Output ONLY valid JSON (no comments, no markdown)"
   ```

3. Add JSON repair in post-processing:
   ```python
   import json_repair
   repaired = json_repair.repair_json(agent_output)
   ```

### Problem: Agent can't find correct recipes

**Solution:**
1. Verify recipe database is populated
2. Try more specific search queries
3. Increase top_k search results:
   ```python
   top_k: int = 10  # Increase from 5
   ```

## 📝 Data Issues

### Problem: Scenarios file is empty or missing

**Solution:**
```bash
# Generate synthetic data
python src/data_utils/create_synthetic_data.py

# Verify file
wc -l fitness_scenarios.jsonl
head -1 fitness_scenarios.jsonl
```

### Problem: Invalid JSON in scenarios file

**Solution:**
```python
# Validate JSONL file
import json
with open('fitness_scenarios.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Line {i}: {e}")
```

## 🔍 Debugging Tips

### Enable verbose logging:

```python
# Add to train.py
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

### Profile memory usage:

```bash
# Monitor during training
python -m memory_profiler train.py
```

### Check model loading:

```python
from src.models import FitnessScenario
scenario = FitnessScenario.model_validate({"step": 1, "scenario": {...}})
print(scenario)
```

### Test individual components:

```bash
# Test agent tools
python -c "from src.agent.tools import create_agent_tools; print(create_agent_tools())"

# Test Pinecone connection
python -c "from src.services import PineconeService; ps = PineconeService(); print(ps)"

# Test reward calculation
python main.py
```

## 📞 Need More Help?

If your issue isn't covered here:

1. **Check logs:**
   ```bash
   tail -f logs/train_*.log
   ```

2. **Enable debug mode:**
   ```bash
   export DEBUG=true
   python train.py
   ```

3. **Create an issue on GitHub** with:
   - Error message (full traceback)
   - Your environment (OS, Python version, CUDA version)
   - Steps to reproduce
   - Logs from `logs/` directory

4. **Common resources:**
   - [OpenPipe ART Docs](https://docs.openpipe.ai/art)
   - [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
   - [Pinecone Docs](https://docs.pinecone.io/)
   - [OpenAI API Docs](https://platform.openai.com/docs)

