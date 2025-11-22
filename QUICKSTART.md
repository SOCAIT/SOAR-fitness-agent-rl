# 🚀 Quick Start Guide

Get up and running with the Fitness Reasoning RL Agent in 5 minutes!

## 📋 Prerequisites

- **Python 3.10+**
- **GPU** (recommended for training)
- **API Keys**: OpenAI + Pinecone

## ⚡ 5-Minute Setup

### Step 1: Clone & Setup (1 min)

```bash
# Clone repository
git clone https://github.com/yourusername/fitness-reasoning-rl-agent.git
cd fitness-reasoning-rl-agent

# Create .env file
cp .env.example .env
```

### Step 2: Add API Keys (1 min)

```bash
# Edit .env and add your keys
nano .env  # or use your favorite editor

# Required:
# OPENAI_API_KEY=sk-your_key
# PINECONE_API_KEY=your_key
```

### Step 3: Install Dependencies (2 min)

```bash
# Automatic setup (handles uv, Python, all deps)
bash train_rl.sh --env-file .env --data-file fitness_scenarios.jsonl
```

**OR** manual setup:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install -e .
pip install openpipe-art[backend,langgraph] langchain-openai pinecone-client weave
```

### Step 4: Test Installation (1 min)

```bash
# Run quick test
python main.py

# Should see reward scores printed
```

## 🎯 What's Next?

### Option A: Train the Agent

```bash
# Full training with shell script
bash train_rl.sh --env-file .env --data-file fitness_scenarios.jsonl --log-dir ./logs

# Or run Python directly
python train.py
```

**Note**: Training takes 30-60 minutes on GPU (T4 or better)

### Option B: Understand the System

Read the architecture section in [README.md](README.md#-how-it-works) to understand:
- How the agent works
- Verifiable rewards system
- RAG-based recipe retrieval

### Option C: Explore the Code

Key files to understand:

```
src/
├── config.py              # Configuration settings
├── agent/
│   ├── rollout.py         # Single agent execution
│   └── tools.py           # Agent tools (recipe search)
├── training/
│   └── trainer.py         # Training loop (GRPO)
└── env/
    └── verifiable_rewards/ # Reward calculation
```

### Option D: Run Example

```bash
# Run example scenario
python main.py

# Output shows:
# - Nutrition plan generated
# - Macro validation score
# - Schema validation score
# - Total reward
```

## 🐛 Troubleshooting

**Problem**: `CUDA out of memory`
```bash
# Reduce batch size in src/config.py
groups_per_step = 1
rollouts_per_group = 2
```

**Problem**: `API key invalid`
```bash
# Verify keys in .env
cat .env

# Test connection
python -c "from src.services import PineconeService; print('OK')"
```

**Problem**: `Module not found`
```bash
# Reinstall dependencies
pip install -e .
pip install openpipe-art[backend,langgraph]>=0.4.11
```

More solutions in [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 📊 Expected Output

When running `python main.py`, you should see:

```
Hello from fitness-reasoning-rl-agent!
Nutrition score: 0.92 info: {'calories_diff': 0.02, 'protein_diff': 0.01}
Nutrition reward score: 0.91 info: {...}
```

## 🎓 Learning Resources

1. **Architecture Deep Dive**: Read [README.md](README.md#-how-it-works)
2. **Common Issues**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
4. **API Docs**: 
   - [OpenPipe ART](https://docs.openpipe.ai/art)
   - [LangGraph](https://langchain-ai.github.io/langgraph/)

## 💡 Tips

- ✅ Start with CPU testing (`enforce_eager=False`) before GPU training
- ✅ Monitor training with Weights & Biases (set `WANDB_API_KEY`)
- ✅ Check GPU memory: `nvidia-smi`
- ✅ View logs: `tail -f logs/train_*.log`

## 🤝 Need Help?

- 📖 Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 🐛 [Open an issue on GitHub](https://github.com/yourusername/fitness-reasoning-rl-agent/issues)
- 💬 Ask in discussions

---

**You're all set! Happy training!** 🏋️✨

