#!/usr/bin/env python3
"""
Prime-RL Fitness Agent Training Script

Fitness Agent training using PrimeIntellect's verifiers library with RAG.
Uses ToolEnv for agentic tool-calling with recipe search.

PREREQUISITES:
--------------
1. Install verifiers: pip install verifiers
2. Install dependencies: uv sync
3. Set up environment variables in .env file:
   - OPENAI_API_KEY (required)
   - PINECONE_API_KEY (required)
   - WANDB_API_KEY (optional, for logging)

USAGE:
------
python scripts/prime-rl/prime-rag-fitnessrl.py

References:
- https://github.com/PrimeIntellect-ai/verifiers
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from pydantic import BaseModel

# Verifiers library
import verifiers as vf
# Note: Rubric is accessed via vf.Rubric, not a separate import

# Pinecone
from pinecone import Pinecone

# W&B for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. Run: pip install wandb")

# Load environment variables
load_dotenv()

# Ensure we're in the project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Validate required environment variables
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file")

if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is required. Please set it in your .env file")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
PROJECT_NAME = "fitness-agent-prime-rl"
SEED = 42
MAX_TURNS = 15

# W&B configuration
WANDB_PROJECT = "fitness-agent-prime-rl"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)  # Your W&B username/team
USE_WANDB = WANDB_AVAILABLE and os.getenv("WANDB_API_KEY")

# Training configuration
TRAINING_CONFIG = {
    "num_examples": 100,
    "rollouts_per_example": 4,
    "max_turns": MAX_TURNS,
}

# ============================================================================
# DATA MODELS
# ============================================================================

class Scenario(BaseModel):
    question: str
    split: Literal["train", "test"]
    id: str
    daily_cal_target: Optional[int] = None
    daily_prot_target: Optional[int] = None
    daily_carb_target: Optional[int] = None
    daily_fat_target: Optional[int] = None
    banned_keywords: Optional[List[str]] = None

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
        answer: JSON string containing the complete meal plan with format:
            {
                "meals": [
                    {"name": "...", "quantity": 1.0, "calories": 500, "proteins": 30, "carbs": 50, "fats": 15, "sequence": 1},
                    ...
                ]
            }
    
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


def extract_final_answer(completion: List[dict]) -> Optional[dict]:
    """Extract the final answer from the completion messages."""
    for msg in reversed(completion):
        content = msg.get("content", "")
        if "Answer submitted:" in content:
            # Extract JSON from the answer
            json_start = content.find("{")
            if json_start != -1:
                return get_payload(content[json_start:])
        # Also check tool calls
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
    """
    Reward for valid JSON schema.
    Returns 1.0 if schema is valid, 0.0 otherwise.
    """
    payload = extract_final_answer(completion)
    if not payload:
        return 0.0
    
    if not isinstance(payload, dict):
        return 0.0
    
    if "meals" not in payload:
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
    """
    Reward for meeting macro targets within ±5% tolerance.
    Returns 1.0 if all macros are within tolerance, 0.0 otherwise.
    """
    payload = extract_final_answer(completion)
    if not payload or "meals" not in payload:
        return 0.0
    
    meals = payload["meals"]
    
    # Get targets from info
    cal_target = info.get("daily_cal_target", 2000)
    prot_target = info.get("daily_prot_target", 150)
    carb_target = info.get("daily_carb_target", 200)
    fat_target = info.get("daily_fat_target", 60)
    
    # Calculate totals
    try:
        total_cal = sum(float(m.get("calories", 0)) for m in meals)
        total_prot = sum(float(m.get("proteins", 0)) for m in meals)
        total_carb = sum(float(m.get("carbs", 0)) for m in meals)
        total_fat = sum(float(m.get("fats", 0)) for m in meals)
    except (TypeError, ValueError):
        return 0.0
    
    tolerance = 0.05
    
    # Check each macro
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
    """
    Reward for meal variety.
    Returns 1.0 if there are at least 3 unique meals.
    """
    payload = extract_final_answer(completion)
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
    """
    Reward for not including banned ingredients.
    Returns 1.0 if no banned keywords are found in meal names.
    """
    payload = extract_final_answer(completion)
    if not payload or "meals" not in payload:
        return 0.0
    
    banned = info.get("banned_keywords", [])
    if not banned:
        return 1.0  # No restrictions
    
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
    
    # Process scenarios
    def process_example(example):
        context = example["context"]
        one_day_prompt = "generate a one day meal plan for user that match its macros and diet"
        
        # Build the question with context
        question = f"{one_day_prompt} Context: {json.dumps(context)}"
        
        # Build the prompt in chat format
        prompt = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        
        # Build info dict for reward functions
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
    
    # Filter to training examples only
    train_examples = [ex for ex in raw_dataset if ex.get("split") == "train"]
    
    processed = [process_example(ex) for ex in train_examples]
    
    return Dataset.from_list(processed)


# ============================================================================
# W&B LOGGING UTILITIES
# ============================================================================

def init_wandb(run_name: str = None):
    """Initialize W&B logging."""
    if not USE_WANDB:
        print("⚠️  W&B logging disabled (no API key or wandb not installed)")
        return None
    
    run_name = run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "model": MODEL_NAME,
            "max_turns": MAX_TURNS,
            "seed": SEED,
            **TRAINING_CONFIG,
        },
        tags=["evaluation", "fitness-agent", "prime-rl"],
    )
    
    print(f"📊 W&B run initialized: {run.url}")
    return run


def log_evaluation_step(step: int, metrics: Dict[str, float], examples: List[Dict] = None):
    """Log evaluation step metrics to W&B."""
    if not USE_WANDB or wandb.run is None:
        return
    
    # Log scalar metrics
    wandb.log({
        "eval/step": step,
        **{f"eval/{k}": v for k, v in metrics.items()},
    }, step=step)
    
    # Log example trajectories as a table
    if examples:
        table = wandb.Table(columns=[
            "scenario_id", "reward", "schema", "macros", "variety", "no_banned",
            "num_turns", "final_answer"
        ])
        
        for ex in examples[:10]:  # Limit to 10 examples
            table.add_data(
                ex.get("scenario_id", ""),
                ex.get("total_reward", 0),
                ex.get("reward_schema", 0),
                ex.get("reward_macros", 0),
                ex.get("reward_variety", 0),
                ex.get("reward_no_banned", 0),
                ex.get("num_turns", 0),
                json.dumps(ex.get("final_answer", {}))[:500],  # Truncate
            )
        
        wandb.log({"eval/examples": table}, step=step)


def log_training_step(step: int, metrics: Dict[str, float]):
    """Log training step metrics to W&B."""
    if not USE_WANDB or wandb.run is None:
        return
    
    wandb.log({
        "train/step": step,
        **{f"train/{k}": v for k, v in metrics.items()},
    }, step=step)


def log_final_results(results: Dict[str, Any]):
    """Log final results and summary to W&B."""
    if not USE_WANDB or wandb.run is None:
        return
    
    # Log summary metrics
    wandb.summary.update({
        "final_mean_reward": results.get("mean_reward", 0),
        "final_schema_rate": results.get("reward_schema_mean", 0),
        "final_macros_rate": results.get("reward_macros_mean", 0),
        "final_variety_rate": results.get("reward_variety_mean", 0),
        "final_no_banned_rate": results.get("reward_no_banned_mean", 0),
    })
    
    # Create reward distribution histogram
    if "rewards" in results:
        wandb.log({
            "eval/reward_distribution": wandb.Histogram(results["rewards"])
        })


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("Prime-RL Fitness Agent Training")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Project: {PROJECT_NAME}")
    print(f"Max Turns: {MAX_TURNS}")
    print(f"W&B Logging: {'Enabled' if USE_WANDB else 'Disabled'}")
    print("=" * 80)
    
    # Initialize W&B
    wandb_run = init_wandb(run_name=f"fitness-eval-{datetime.now().strftime('%H%M')}")
    
    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = load_fitness_dataset()
    print(f"   Loaded {len(dataset)} training examples")
    
    if USE_WANDB:
        wandb.config.update({"dataset_size": len(dataset)})
    
    # Create rubric with reward functions
    print("\n🎯 Creating rubric...")
    rubric = vf.Rubric(
        funcs=[
            reward_schema,    # Schema validation
            reward_macros,    # Macro accuracy
            reward_variety,   # Meal variety
            reward_no_banned, # No banned ingredients
        ],
        weights=[0.2, 0.5, 0.15, 0.15],  # Macro accuracy is most important
    )
    
    # Create ToolEnv
    print("\n🔧 Creating ToolEnv...")
    env = vf.ToolEnv(
        dataset=dataset,
        rubric=rubric,
        tools=[recipe_semantic_search, return_final_answer],
        max_turns=MAX_TURNS,
    )
    
    # Run evaluation (for testing before full training)
    print("\n🧪 Running evaluation...")
    from openai import AsyncOpenAI
    
    # Use local vLLM endpoint or OpenAI
    client = AsyncOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )
    
    import asyncio
    
    num_eval_examples = min(10, len(dataset))
    rollouts_per_example = 2
    
    async def run_eval():
        results = await env.evaluate(
            client=client,
            model=MODEL_NAME,
            num_examples=num_eval_examples,
            rollouts_per_example=rollouts_per_example,
        )
        return results
    
    print(f"   Evaluating {num_eval_examples} examples with {rollouts_per_example} rollouts each...")
    results = asyncio.run(run_eval())
    
    # Extract metrics
    mean_reward = results.get('mean_reward', 0)
    schema_rate = results.get('reward_schema_mean', 0)
    macros_rate = results.get('reward_macros_mean', 0)
    variety_rate = results.get('reward_variety_mean', 0)
    no_banned_rate = results.get('reward_no_banned_mean', 0)
    
    # Print results
    print("\n📊 Evaluation Results:")
    print(f"   Mean Reward: {mean_reward:.3f}")
    print(f"   Schema Pass Rate: {schema_rate:.1%}")
    print(f"   Macro Pass Rate: {macros_rate:.1%}")
    print(f"   Variety Pass Rate: {variety_rate:.1%}")
    print(f"   No Banned Pass Rate: {no_banned_rate:.1%}")
    
    # Log to W&B
    log_evaluation_step(
        step=0,
        metrics={
            "mean_reward": mean_reward,
            "schema_rate": schema_rate,
            "macros_rate": macros_rate,
            "variety_rate": variety_rate,
            "no_banned_rate": no_banned_rate,
            "num_examples": num_eval_examples,
            "rollouts_per_example": rollouts_per_example,
        }
    )
    log_final_results(results)
    
    # Convert to HF dataset format for training
    print("\n💾 Creating training dataset...")
    train_dataset = env.make_dataset(results)
    print(f"   Created {len(train_dataset)} training examples")
    
    # Finish W&B run
    if USE_WANDB and wandb.run is not None:
        wandb.finish()
        print("\n📊 W&B run finished. View results at the link above.")
    
    # For full training, use prime-rl trainer
    print("\n🚀 To run full training with prime-rl:")
    print("   1. Start vLLM server with your model")
    print("   2. Configure accelerate/deepspeed")
    print("   3. Use verifiers.trainers.GRPOTrainer")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

