#!/usr/bin/env python3
"""
Prime-RL GRPO Training Script

Full training loop using TRL's GRPO (Group Relative Policy Optimization) with LoRA.

PREREQUISITES:
--------------
1. Install dependencies:
   pip install trl peft transformers accelerate wandb

2. Run training:
   accelerate launch scripts/prime-rl/train_grpo.py

NOTE: vLLM is NOT required for training. The GRPOTrainer handles generation internally.
      vLLM is only used for the separate evaluation script (prime-rag-fitnessrl.py).

MEMORY REQUIREMENTS:
-------------------
- With LoRA (default): ~20-30GB GPU memory for 14B model
- Without LoRA: 80GB+ GPU memory for 14B model

References:
- https://huggingface.co/docs/trl/main/en/grpo_trainer
- https://github.com/PrimeIntellect-ai/verifiers
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from pydantic import BaseModel

# Verifiers library (for environments and rubrics)
import verifiers as vf

# TRL for GRPO training
from trl import GRPOTrainer, GRPOConfig

# Transformers for callbacks
from transformers import TrainerCallback

# PEFT for LoRA
from peft import LoraConfig, TaskType

# Pinecone
from pinecone import Pinecone

# W&B for comprehensive logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run: pip install wandb")

# Load environment variables
load_dotenv()

# Project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Validate environment
if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is required")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
# Note: vLLM is NOT needed for training - only for separate evaluation script

# W&B configuration
WANDB_PROJECT = "fitness-agent-prime-rl"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
USE_WANDB = WANDB_AVAILABLE and os.getenv("WANDB_API_KEY")
RUN_NAME = f"grpo-fitness-{datetime.now().strftime('%Y%m%d-%H%M')}"

MAX_TURNS = 15

# ============================================================================
# W&B LOGGING UTILITIES
# ============================================================================

class CompletionLogger:
    """Logs LLM completions for qualitative analysis."""
    
    def __init__(self, log_dir: Path, log_to_wandb: bool = True):
        self.log_dir = log_dir
        self.log_to_wandb = log_to_wandb
        self.completions_file = log_dir / "completions.jsonl"
        self.completions_file.parent.mkdir(parents=True, exist_ok=True)
        self.sample_completions = []  # Store samples for W&B
        self.max_samples = 50  # Keep last N samples for W&B
        
    def log_completions(
        self, 
        step: int,
        completions: List[str],
        prompts: List[str] = None,
        rewards: List[float] = None,
        infos: List[Dict] = None,
    ):
        """Log completions to file and W&B."""
        if not completions:
            return
        
        # Save to JSONL file
        with open(self.completions_file, "a", encoding="utf-8") as f:
            for i, completion in enumerate(completions):
                record = {
                    "step": step,
                    "completion": completion,
                    "reward": rewards[i] if rewards else None,
                    "prompt_preview": prompts[i][:200] if prompts and i < len(prompts) else None,
                    "scenario_id": infos[i].get("scenario_id") if infos and i < len(infos) else None,
                    "timestamp": datetime.now().isoformat(),
                }
                f.write(json.dumps(record) + "\n")
        
        # Store samples for W&B (keep last N)
        for i, completion in enumerate(completions[:3]):  # Log first 3 per batch
            sample = {
                "step": step,
                "completion": completion[:1000],  # Truncate for W&B
                "reward": rewards[i] if rewards and i < len(rewards) else None,
                "scenario_id": infos[i].get("scenario_id") if infos and i < len(infos) else None,
            }
            self.sample_completions.append(sample)
            
            # Keep only last N samples
            if len(self.sample_completions) > self.max_samples:
                self.sample_completions.pop(0)
        
        # Print example to console every N steps
        if step % 10 == 0 and completions:
            print(f"\n{'='*80}")
            print(f"📝 Sample Completion at Step {step}")
            print(f"{'='*80}")
            print(f"Reward: {rewards[0] if rewards else 'N/A'}")
            if infos and infos[0]:
                print(f"Scenario ID: {infos[0].get('scenario_id', 'N/A')}")
            print(f"\nCompletion Preview (first 500 chars):")
            print("-" * 80)
            print(completions[0][:500])
            print("..." if len(completions[0]) > 500 else "")
            print(f"{'='*80}\n")
    
    def log_to_wandb(self, step: int):
        """Log sample completions to W&B as a table."""
        if not self.log_to_wandb or not USE_WANDB or wandb.run is None:
            return
        
        if not self.sample_completions:
            return
        
        # Create W&B table
        table = wandb.Table(columns=["step", "scenario_id", "reward", "completion"])
        for sample in self.sample_completions[-20:]:  # Last 20 samples
            table.add_data(
                sample["step"],
                sample.get("scenario_id", "N/A"),
                sample.get("reward", 0.0),
                sample["completion"],
            )
        
        wandb.log({"train/completions": table}, step=step)


class GRPOTrainingCallback(TrainerCallback):
    """Callback to ensure proper W&B logging during GRPO training."""
    
    def __init__(self, custom_logger, completion_logger: CompletionLogger = None):
        self.custom_logger = custom_logger
        self.completion_logger = completion_logger
        self.last_logged_step = -1
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics - Transformers handles W&B logging automatically.
        
        This callback is mainly for debugging and ensuring consistency.
        With report_to="wandb", Transformers logs metrics with global_step automatically.
        """
        global _current_step
        
        if logs is None or not USE_WANDB or wandb.run is None:
            return
        
        # Get the current global step (cumulative across all epochs)
        global_step = state.global_step if hasattr(state, 'global_step') else 0
        _current_step = global_step  # Update global step for reward function
        
        # Transformers automatically logs via report_to="wandb" with global_step
        # We don't need to log again, but we can verify what's being logged
        if global_step > self.last_logged_step:
            self.last_logged_step = global_step
            
            # Log completions to W&B periodically
            if self.completion_logger and global_step % 10 == 0:
                self.completion_logger.log_to_wandb(global_step)
            
            # Debug: print what Transformers is logging
            if global_step <= 5:
                print(f"📊 Trainer logged at global_step {global_step}: {list(logs.keys())}")
                # Note: Transformers logs these with step=global_step automatically
                # Metrics include: loss, learning_rate, grad_norm, epoch, etc.


class FitnessAgentWandbLogger:
    """Custom W&B logger for fitness agent training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.step = 0
        self.eval_step = 0
        
    def init_run(self) -> Optional[Any]:
        """Initialize W&B run with full configuration."""
        if not USE_WANDB:
            print("⚠️  W&B logging disabled")
            return None
        
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=RUN_NAME,
            config={
                "model": MODEL_NAME,
                "max_turns": MAX_TURNS,
                "training_method": "GRPO + LoRA",
                **self.config,
            },
            tags=["training", "grpo", "fitness-agent", "prime-rl"],
            group="grpo-training",
        )
        
        # Define custom metrics for better visualization
        # Use global_step as the primary step metric (matches Transformers default)
        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step")
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        wandb.define_metric("rollout/step")
        wandb.define_metric("rollout/*", step_metric="rollout/step")
        
        print(f"📊 W&B run initialized: {run.url}")
        return run
    
    def log_training_step(self, metrics: Dict[str, float]):
        """Log training step metrics."""
        if not USE_WANDB or wandb.run is None:
            return
        
        self.step += 1
        wandb.log({
            "train/step": self.step,
            **{f"train/{k}": v for k, v in metrics.items()},
        }, step=self.step)
    
    def log_rollout_batch(
        self, 
        batch_idx: int, 
        rewards: List[float],
        reward_breakdown: Dict[str, List[float]],
        num_turns: List[int],
    ):
        """Log rollout batch metrics with histograms."""
        if not USE_WANDB or wandb.run is None:
            return
        
        import numpy as np
        
        log_data = {
            "rollout/step": batch_idx,
            "rollout/mean_reward": np.mean(rewards),
            "rollout/max_reward": np.max(rewards),
            "rollout/min_reward": np.min(rewards),
            "rollout/std_reward": np.std(rewards),
            "rollout/mean_turns": np.mean(num_turns),
            "rollout/reward_histogram": wandb.Histogram(rewards),
            "rollout/turns_histogram": wandb.Histogram(num_turns),
        }
        
        # Add breakdown by reward type
        for reward_name, values in reward_breakdown.items():
            log_data[f"rollout/{reward_name}_mean"] = np.mean(values)
            log_data[f"rollout/{reward_name}_rate"] = sum(1 for v in values if v > 0.5) / len(values)
        
        wandb.log(log_data, step=batch_idx)
    
    def log_evaluation(
        self, 
        results: Dict[str, Any],
        examples: List[Dict] = None,
    ):
        """Log evaluation results with detailed metrics."""
        if not USE_WANDB or wandb.run is None:
            return
        
        self.eval_step += 1
        
        log_data = {
            "eval/step": self.eval_step,
            "eval/mean_reward": results.get("mean_reward", 0),
            "eval/schema_rate": results.get("reward_schema_mean", 0),
            "eval/macros_rate": results.get("reward_macros_mean", 0),
            "eval/variety_rate": results.get("reward_variety_mean", 0),
            "eval/no_banned_rate": results.get("reward_no_banned_mean", 0),
        }
        
        # Add reward distribution if available
        if "rewards" in results:
            log_data["eval/reward_distribution"] = wandb.Histogram(results["rewards"])
        
        wandb.log(log_data, step=self.eval_step)
        
        # Log example trajectories as a table
        if examples:
            self._log_example_table(examples)
    
    def _log_example_table(self, examples: List[Dict]):
        """Log example trajectories as a W&B table."""
        if not USE_WANDB or wandb.run is None:
            return
        
        columns = [
            "scenario_id", "total_reward", "schema", "macros", "variety", 
            "no_banned", "num_turns", "final_answer_preview"
        ]
        
        table = wandb.Table(columns=columns)
        
        for ex in examples[:20]:  # Limit to 20 examples
            final_answer = ex.get("final_answer", {})
            preview = json.dumps(final_answer)[:300] if final_answer else ""
            
            table.add_data(
                ex.get("scenario_id", ""),
                ex.get("total_reward", 0),
                ex.get("reward_schema", 0),
                ex.get("reward_macros", 0),
                ex.get("reward_variety", 0),
                ex.get("reward_no_banned", 0),
                ex.get("num_turns", 0),
                preview,
            )
        
        wandb.log({"eval/examples": table}, step=self.eval_step)
    
    def log_model_checkpoint(self, checkpoint_path: str, step: int):
        """Log model checkpoint as artifact."""
        if not USE_WANDB or wandb.run is None:
            return
        
        artifact = wandb.Artifact(
            name=f"model-checkpoint-{step}",
            type="model",
            description=f"Model checkpoint at step {step}",
        )
        artifact.add_dir(checkpoint_path)
        wandb.log_artifact(artifact)
    
    def log_final_summary(self, final_metrics: Dict[str, Any]):
        """Log final training summary."""
        if not USE_WANDB or wandb.run is None:
            return
        
        # Update W&B summary with final metrics
        wandb.summary.update({
            f"final/{k}": v for k, v in final_metrics.items()
        })
        
        # Create a summary bar chart
        if "reward_breakdown" in final_metrics:
            breakdown = final_metrics["reward_breakdown"]
            data = [[k, v] for k, v in breakdown.items()]
            table = wandb.Table(data=data, columns=["metric", "value"])
            wandb.log({
                "final/reward_breakdown": wandb.plot.bar(
                    table, "metric", "value", title="Final Reward Breakdown"
                )
            })
    
    def finish(self):
        """Finish W&B run."""
        if USE_WANDB and wandb.run is not None:
            wandb.finish()
            print("📊 W&B run finished.")

# ============================================================================
# PINECONE SETUP
# ============================================================================

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
recipe_index = pc.Index("syntrafit-meals-nutrition")

def extract_meal_names(data):
    """Extract meal information from Pinecone results."""
    data = data['result']
    return [
        {
            "id": hit['_id'],
            "name": hit["fields"]["name"],
            "calories": hit["fields"]["calories"],
            "carbs": hit["fields"]["carbs"],
            "protein": hit["fields"]["proteins"],
            "fat": hit["fields"]["fats"]
        }
        for hit in data["hits"]
        if "fields" in hit and "name" in hit["fields"]
    ]

# ============================================================================
# TOOLS
# ============================================================================

def recipe_semantic_search(meal_query: str, k: int = 5) -> str:
    """
    Search the recipe database for recipes matching the query.
    
    Args:
        meal_query: Natural language description of the meal to search for
        k: Number of results to return (default 5)
    
    Returns:
        JSON string containing list of matching recipes with their macros
    """
    try:
        results = recipe_index.search(
            namespace="syntrafit",
            query={"top_k": k, "inputs": {'text': meal_query}}
        )
        res_list = extract_meal_names(results)
        return json.dumps(res_list)
    except Exception as e:
        return json.dumps({"error": str(e)})


def return_final_answer(answer: str) -> str:
    """
    Return the final meal plan answer.
    
    Args:
        answer: JSON string containing the complete meal plan
    
    Returns:
        Confirmation of the submitted answer
    """
    return f"Answer submitted: {answer}"

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

PLANNER_SYSTEM_PROMPT = """You are a nutrition planner specialist who creates daily nutrition plans. Think carefully and pay great attention to macro numbers.

You must create a one-day meal plan that meets the user's macro and dietary targets.

TOOLS YOU CAN USE:
1) recipe_semantic_search(meal_query, k) – Search for relevant recipes and return their true macros.
2) return_final_answer(answer) – Return the final answer (JSON plan).

EXECUTION POLICY:
- Search for recipes to find meals with accurate macro information.
- Calculate quantities to meet the user's daily macro targets (±5% tolerance).
- The FINAL step must be a call to return_final_answer with the complete JSON meal plan.

============================================================
NUTRITION PLAN PIPELINE
============================================================

1) ANALYZE TARGETS
   • Extract the user's daily macro targets: calories, protein, carbs, fat.
   • Note any banned ingredients or dietary restrictions.

2) SEARCH FOR RECIPES
   • Use recipe_semantic_search to find suitable recipes for breakfast, lunch, dinner, and snacks.
   • You MUST use the macros returned by the tool, NOT your own estimates.

3) CALCULATE QUANTITIES
   • For each meal, set a "quantity" multiplier to scale the macros.
   • Example: If a recipe has 400 cal and you need 600 cal from it, use quantity=1.5
   • Sum all meals to verify totals are within ±5% of targets.

4) BUILD FINAL JSON
   • Structure:
     {
       "meals": [
         {
           "name": "Recipe Name from Search",
           "quantity": 1.5,
           "calories": 600,
           "proteins": 45,
           "carbs": 60,
           "fats": 20,
           "sequence": 1
         }
       ]
     }

5) SUBMIT ANSWER
   • Call return_final_answer with the JSON meal plan.
"""

# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def _extract_first_json_segment(s: str) -> Optional[str]:
    """Extract the first valid JSON object from a string."""
    start_candidates = [s.find('{'), s.find('[')]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    depth_obj, depth_arr, in_str, esc = 0, 0, False, False
    opener = s[start]
    want_obj, want_arr = opener == '{', opener == '['
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth_obj += 1
            elif ch == '}':
                depth_obj -= 1
            elif ch == '[':
                depth_arr += 1
            elif ch == ']':
                depth_arr -= 1
            if want_obj and depth_obj == 0 and depth_arr == 0 and i >= start:
                return s[start:i+1]
            if want_arr and depth_arr == 0 and depth_obj == 0 and i >= start:
                return s[start:i+1]
    return None


def get_payload(obj) -> dict:
    """Extract payload dictionary from various input formats."""
    if hasattr(obj, "answer"):
        obj = obj.answer
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            seg = _extract_first_json_segment(obj)
            if seg:
                try:
                    obj = json.loads(seg)
                except json.JSONDecodeError:
                    pass
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"_root": obj}
    return {}


def extract_final_answer_from_completion(completion: List[dict]) -> Optional[dict]:
    """Extract the final answer from the completion messages."""
    for msg in reversed(completion):
        content = msg.get("content", "")
        if "Answer submitted:" in content:
            json_start = content.find("{")
            if json_start != -1:
                return get_payload(content[json_start:])
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            if tc.get("function", {}).get("name") == "return_final_answer":
                args = tc.get("function", {}).get("arguments", "{}")
                try:
                    parsed = json.loads(args)
                    if "answer" in parsed:
                        return get_payload(parsed["answer"])
                except json.JSONDecodeError:
                    pass
    return None


def reward_schema(prompt, completion, info) -> float:
    """Reward for valid JSON schema."""
    payload = extract_final_answer_from_completion(completion)
    if not payload or "meals" not in payload:
        return 0.0
    
    meals = payload["meals"]
    if not isinstance(meals, list) or len(meals) == 0:
        return 0.0
    
    required_keys = ["name", "quantity", "calories", "proteins", "carbs", "fats"]
    for meal in meals:
        if not isinstance(meal, dict):
            return 0.0
        for k in required_keys:
            if k not in meal:
                return 0.0
    
    return 1.0


def reward_macros(prompt, completion, info) -> float:
    """Reward for meeting macro targets within ±5% tolerance."""
    payload = extract_final_answer_from_completion(completion)
    if not payload or "meals" not in payload:
        return 0.0
    
    meals = payload["meals"]
    
    cal_target = info.get("daily_cal_target", 2000)
    prot_target = info.get("daily_prot_target", 150)
    carb_target = info.get("daily_carb_target", 200)
    fat_target = info.get("daily_fat_target", 60)
    
    try:
        total_cal = sum(float(m.get("calories", 0)) for m in meals)
        total_prot = sum(float(m.get("proteins", 0)) for m in meals)
        total_carb = sum(float(m.get("carbs", 0)) for m in meals)
        total_fat = sum(float(m.get("fats", 0)) for m in meals)
    except (TypeError, ValueError):
        return 0.0
    
    tolerance = 0.05
    
    if cal_target > 0 and abs(total_cal - cal_target) / cal_target > tolerance:
        return 0.0
    if prot_target > 0 and abs(total_prot - prot_target) / prot_target > tolerance:
        return 0.0
    if carb_target > 0 and abs(total_carb - carb_target) / carb_target > tolerance:
        return 0.0
    if fat_target > 0 and abs(total_fat - fat_target) / fat_target > tolerance:
        return 0.0
    
    return 1.0


def reward_variety(prompt, completion, info) -> float:
    """Reward for meal variety (at least 3 unique meals)."""
    payload = extract_final_answer_from_completion(completion)
    if not payload or "meals" not in payload:
        return 0.0
    
    meals = payload["meals"]
    if len(meals) < 3:
        return 0.0
    
    unique_names = set(m.get("name", "").lower().strip() for m in meals)
    if len(unique_names) < 3:
        return 0.0
    
    return 1.0


def reward_no_banned(prompt, completion, info) -> float:
    """Reward for not including banned ingredients."""
    payload = extract_final_answer_from_completion(completion)
    if not payload or "meals" not in payload:
        return 0.0
    
    banned = info.get("banned_keywords", [])
    if not banned:
        return 1.0
    
    meals = payload["meals"]
    for meal in meals:
        name = meal.get("name", "").lower()
        for banned_kw in banned:
            if banned_kw.lower() in name:
                return 0.0
    
    return 1.0


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def load_fitness_dataset() -> Dataset:
    """Load and prepare the fitness scenarios dataset."""
    dataset_path = project_root / "data" / "fitness_scenarios.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    raw_dataset = load_dataset("json", data_files=str(dataset_path))["train"]
    
    def process_example(example):
        context = example["context"]
        one_day_prompt = "generate a one day meal plan for user that match its macros and diet"
        question = f"{one_day_prompt} Context: {json.dumps(context)}"
        
        prompt = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        info = {
            "daily_cal_target": context.get("daily_cal_target"),
            "daily_prot_target": context.get("daily_prot_target"),
            "daily_carb_target": context.get("daily_carb_target"),
            "daily_fat_target": context.get("daily_fat_target"),
            "banned_keywords": context.get("banned_keywords", []),
            "scenario_id": example["id"],
        }
        
        return {
            "prompt": prompt,
            "info": info,
            "task": "nutrition_planning",
        }
    
    train_examples = [ex for ex in raw_dataset if ex.get("split") == "train"]
    processed = [process_example(ex) for ex in train_examples]
    
    return Dataset.from_list(processed)


# ============================================================================
# REWARD FUNCTION FOR TRL
# ============================================================================

# Global completion logger (set in main())
_completion_logger: Optional[CompletionLogger] = None
_current_step: int = 0

def fitness_reward(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Compute rewards for a batch of completions.
    This is called by GRPOTrainer during training.
    
    Args:
        completions: List of model-generated completions (strings)
        prompts: List of prompts (optional)
        **kwargs: Additional info including 'infos' list
    
    Returns:
        List of reward values
    """
    global _completion_logger, _current_step
    
    rewards = []
    infos = kwargs.get("infos", [{}] * len(completions))
    
    for i, completion in enumerate(completions):
        info = infos[i] if i < len(infos) else {}
        prompt = prompts[i] if prompts and i < len(prompts) else ""
        
        # Parse completion as messages
        completion_msgs = [{"role": "assistant", "content": completion}]
        
        # Calculate individual rewards
        r_schema = reward_schema(prompt, completion_msgs, info)
        r_macros = reward_macros(prompt, completion_msgs, info)
        r_variety = reward_variety(prompt, completion_msgs, info)
        r_banned = reward_no_banned(prompt, completion_msgs, info)
        
        # Weighted sum
        total = 0.2 * r_schema + 0.5 * r_macros + 0.15 * r_variety + 0.15 * r_banned
        rewards.append(total)
    
    # Log completions if logger is available
    if _completion_logger is not None:
        prompt_strings = [
            json.dumps(p) if isinstance(p, (list, dict)) else str(p) 
            for p in (prompts or [])
        ]
        _completion_logger.log_completions(
            step=_current_step,
            completions=completions,
            prompts=prompt_strings,
            rewards=rewards,
            infos=infos,
        )
    
    return rewards


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("Prime-RL GRPO Training with LoRA")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Training Method: GRPO + LoRA (parameter-efficient)")
    print(f"W&B Logging: {'Enabled' if USE_WANDB else 'Disabled'}")
    print("=" * 80)
    
    # Initialize W&B logger BEFORE creating trainer
    # This ensures the trainer's built-in W&B logging uses our run
    wandb_logger = FitnessAgentWandbLogger(config={
        "model": MODEL_NAME,
        "max_turns": MAX_TURNS,
        "num_generations": 2,
        "max_completion_length": 1024,
        "lora_rank": 16,
        "lora_alpha": 32,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
    })
    wandb_run = wandb_logger.init_run()
    
    # Initialize completion logger for qualitative analysis
    output_dir = Path("./outputs/prime-rl-fitness")
    completion_logger = CompletionLogger(
        log_dir=output_dir,
        log_to_wandb=USE_WANDB,
    )
    global _completion_logger
    _completion_logger = completion_logger
    
    print(f"\n📝 Completion logging enabled:")
    print(f"   - Completions saved to: {completion_logger.completions_file}")
    print(f"   - W&B table: train/completions (updated every 10 steps)")
    print(f"   - Console preview: every 10 steps")
    
    # Ensure W&B is initialized for trainer
    if USE_WANDB:
        # Set environment variable so trainer uses existing run
        os.environ["WANDB_RUN_ID"] = str(wandb.run.id) if wandb.run else ""
    
    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = load_fitness_dataset()
    print(f"   Loaded {len(dataset)} training examples")
    
    if USE_WANDB and wandb.run:
        wandb.config.update({"dataset_size": len(dataset)})
    
    # Create GRPO Trainer with simple API
    # See: https://huggingface.co/docs/trl/main/en/grpo_trainer
    print("\n🏋️ Creating GRPO Trainer...")
    
    # Memory-efficient training config
    # For 14B model on 80GB GPU, use small batches + gradient accumulation
    training_config = GRPOConfig(
        output_dir="./outputs/prime-rl-fitness",
        num_train_epochs=3,
        
        # Memory-efficient settings
        per_device_train_batch_size=1,      # Small batch for large model
        gradient_accumulation_steps=16,     # Accumulate for effective batch of 16
        
        # GRPO specific - reduce generations for memory
        num_generations=2,                  # Reduced from 4
        max_completion_length=1024,         # Reduced from 2048
        
        # Mixed precision for memory savings
        bf16=True,                          # Use bf16 on Ampere+ GPUs
        
        # Gradient checkpointing saves memory
        gradient_checkpointing=True,
        
        # Learning rate
        learning_rate=1e-5,
        warmup_ratio=0.1,
        
        # Logging
        logging_steps=1,
        save_steps=100,
        report_to="wandb" if USE_WANDB else "none",
        run_name=RUN_NAME,
        
        # Reduce memory fragmentation
        dataloader_pin_memory=False,
    )
    
    print("   Memory-efficient settings enabled:")
    print(f"   - Batch size: 1 (effective: 16 with grad accum)")
    print(f"   - Gradient checkpointing: enabled")
    print(f"   - Mixed precision: bf16")
    print(f"   - Num generations: 2")
    
    # LoRA configuration for parameter-efficient training
    # This drastically reduces memory usage by only training ~0.1% of parameters
    peft_config = LoraConfig(
        r=16,                               # LoRA rank (higher = more capacity, more memory)
        lora_alpha=32,                      # LoRA alpha (scaling factor)
        lora_dropout=0.05,                  # Dropout for regularization
        target_modules=[                    # Modules to apply LoRA to
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    print("   LoRA configuration:")
    print(f"   - Rank (r): {peft_config.r}")
    print(f"   - Alpha: {peft_config.lora_alpha}")
    print(f"   - Target modules: {len(peft_config.target_modules)} layers")
    print(f"   - Trainable params: ~0.1% of full model")
    
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=fitness_reward,
        train_dataset=dataset,
        args=training_config,
        peft_config=peft_config,            # Enable LoRA
        callbacks=[GRPOTrainingCallback(wandb_logger, completion_logger)] if USE_WANDB else None,
    )
    
    # Train with logging
    print("\n🚀 Starting training...")
    print("   View training progress at: https://wandb.ai" if USE_WANDB else "")
    
    try:
        trainer.train()
        
        # Log final metrics
        final_metrics = {
            "total_steps": trainer.state.global_step if hasattr(trainer, 'state') else 0,
            "epochs_completed": 3,
        }
        wandb_logger.log_final_summary(final_metrics)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if USE_WANDB and wandb.run:
            wandb.mark_preempting()
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        if USE_WANDB and wandb.run:
            wandb.alert(
                title="Training Error",
                text=str(e),
                level=wandb.AlertLevel.ERROR,
            )
        raise
    finally:
        wandb_logger.finish()
    
    # Save final model
    print("\n💾 Saving model...")
    trainer.save_model()
    
    # Log model as artifact
    if USE_WANDB:
        wandb_logger.log_model_checkpoint(
            "./outputs/prime-rl-fitness", 
            trainer.state.global_step if hasattr(trainer, 'state') else 0
        )
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: ./outputs/prime-rl-fitness")
    if wandb_run:
        print(f"   W&B run: {wandb_run.url}")


if __name__ == "__main__":
    main()

