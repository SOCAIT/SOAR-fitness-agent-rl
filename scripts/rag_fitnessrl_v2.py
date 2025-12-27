#!/usr/bin/env python3
"""
RAG-FitnessRL-v2 Training Script

Fitness Agent training using ART with LangGraph and RAG.
Refined reward function including:
1. Strict Macro Verification (+/- 5%)
2. Schema & Format Validation
3. Variety & Meal Count Checks (Heuristic + LLM Judge)

PREREQUISITES:
--------------
1. Install dependencies: `uv sync`
2. Set up environment variables in .env file:
   - OPENAI_API_KEY (required)
   - PINECONE_API_KEY (required)
   - WANDB_API_KEY (optional, for logging)
"""

## CONFIGURATION

BASE_MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
MODEL_NAME = "fitness-agent-langgraph-32B-qwen2.5-001"
PROJECT_NAME = "fitness-agent-langgraph-rag-v2"
SEED = 42
TENSOR_PARALLEL_SIZE = 2
GPU_MEMORY_UTILIZATION = 0.60
MAX_SEQ_LENGTH = 8192
ENFORCE_EAGER = True

# Import unsloth FIRST before any other ML libraries to avoid circular import issues
# This is critical for tensor parallelism with vLLM worker spawning
import os
import sys

# Import unsloth early to ensure it's fully initialized before vLLM spawns workers
# This prevents circular import errors during multiprocessing worker initialization
# The import must happen before any transformers/trl/peft imports
try:
    import unsloth  # noqa: F401
except ImportError:
    pass  # Unsloth may not be available, but that's okay
import json
import math
import re
import random
import difflib
import uuid
from pathlib import Path
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functools import wraps
from statistics import median

# Third-party imports
from dotenv import load_dotenv
from datasets import Dataset, Features, Sequence, Value, load_dataset
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm
from tenacity import retry, stop_after_attempt

# LangChain and LangGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ART and related
import art
from art.local import LocalBackend
from art.langgraph import init_chat_model

# LiteLLM and Weave
from litellm import acompletion
import weave

# Pinecone
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Ensure we're in the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Validate required environment variables
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file")

if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is required. Please set it in your .env file")

if not os.environ.get("WANDB_API_KEY"):
    print("⚠️  WANDB_API_KEY is not set. Skipping W&B logging.")

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


class FinalAnswer(BaseModel):
    answer: Dict[str, Any]

    @field_validator("answer", mode="before")
    @classmethod
    def ensure_dict(cls, v):
            if isinstance(v, FinalAnswer):
                return v.answer
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError as e:
                    raise ValueError(f"answer must be a JSON object string or dict; got invalid JSON: {e}")
            if isinstance(v, dict):
                return v
            raise TypeError(f"Unsupported type for answer: {type(v).__name__}")


@dataclass
class SearchResult:
    message_id: str
    snippet: str

# ============================================================================
# DATA LOADING
# ============================================================================

dataset_path = project_root / "data" / "fitness_scenarios.jsonl"
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

dataset = load_dataset("json", data_files=str(dataset_path))
training_scenarios = dataset["train"]

print("Dataset loaded successfully!")

# Processing functions
def one_day_meal_question(example):
    one_day_prompt = "generate a one day meal plan for user that match its macros and diet"
    example['input_question'] = f"{one_day_prompt}"
    return example

def combine_question_and_context(example):
    context_str = json.dumps(example['context'])
    example['question'] = f"{example['input_question']} Context: {context_str}"
    return example

def convert_val_to_test(example):
    if example['split'] == 'val':
       example['split'] = 'test'
    return example

def get_target_nutrition_data(example):
    example['daily_cal_target'] = example['context']['daily_cal_target']
    example['daily_prot_target'] = example['context']['daily_prot_target']
    example['daily_carb_target'] = example['context']['daily_carb_target']
    example['daily_fat_target'] = example['context']['daily_fat_target']
    example['banned_keywords'] = example['context']['banned_keywords']
    return example

training_scenarios = training_scenarios.map(one_day_meal_question)
training_scenarios = training_scenarios.map(combine_question_and_context)
training_scenarios = training_scenarios.map(convert_val_to_test)
training_scenarios = training_scenarios.map(get_target_nutrition_data)

scenarios_list = []
for example in training_scenarios:
    scenario = Scenario(
        question=example['question'],
        split=example['split'],
        id=str(example['id']),
        daily_cal_target=example['daily_cal_target'],
        daily_prot_target=example['daily_prot_target'],
        daily_carb_target=example['daily_carb_target'],
        daily_fat_target=example['daily_fat_target'],
        banned_keywords=example['banned_keywords']
    )
    scenarios_list.append(scenario)

print(f"Created a list of {len(scenarios_list)} Scenario objects.")

# ============================================================================
# MODEL CONFIG
# ============================================================================

# Model configuration is set at the top of the file (lines 22-24)
# BASE_MODEL_NAME, MODEL_NAME, and PROJECT_NAME are already defined above

model = None
backend = None

def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    print(f"🌱 All seeds set to: {seed}")

# ============================================================================
# PROMPTS
# ============================================================================

VARIETY_JUDGE_PROMPT = """
You are a nutrition expert evaluating a daily meal plan.
Evaluate the plan based on Variety and Meal Frequency.

Criteria:
1. Variety: Are there different types of foods? (Not just Chicken & Rice 5 times).
2. Meal Frequency: Are there at least 3 distinct meals/snacks?
3. Practicality: Is the plan realistic for a human to eat in one day?

Return a JSON with:
{
  "score": float, // 0.0 to 1.0, where 1.0 is excellent variety and structure
  "reason": "string explanation"
}
"""

PLANNER_PROMPT = f"""
You are a nutrition planner specialist who creates daily nutrition plans. Think carefully and pay great attention to macro numbers.

You must create a one-day meal plan that meets the user's macro and dietary targets.

TOOLS YOU CAN USE (names must match exactly):
1) recipe_semantic_search(meal_query, k) – Search for relevant recipes and return their true macros.
2) return_final_answer_tool(answer) – Return the final answer (JSON plan).

ROUTING RULES (VERY IMPORTANT):
- If the user asks for a meal/nutrition plan, you MUST NOT call get_available_exercises or generate_workout_plan_mutation.
- Even if the request is for a 7-day plan, create only a single-day meal plan.

EXECUTION POLICY:
- You MAY call tools during reasoning to gather information and choose recipes.
- The FINAL assistant message must output ONLY a single call to return_final_answer_tool with the exact JSON plan (stringified if needed).
- You may take up to 20 turns to find the answer.

============================================================
NUTRITION PLAN PIPELINE
============================================================

1) PLAN SKELETON
   • Generate a one-day meal plan for the user. The plan should have meals that fulfill the user's daily macro targets.
   • Create a reasonable number of meals (at least 3 distinct meals/snacks).
   • Meal names in the final plan MUST be recipe names returned by recipe_semantic_search (do not invent names).

2) USE RECIPE SEMANTIC SEARCH TOOL
   • Use recipe_semantic_search to find relevant recipes with correct nutrition info.
   • You MUST ALWAYS use the macros retrieved from the tool and NOT infer your own data.
   • For each meal you include:
     - Set "name" to the exact recipe name from the tool
     - Set "quantity" to a multiplier (e.g., 1.0 for one serving, 1.5 for 1.5 servings, 0.5 for half serving)
     - Calculate macros as: quantity × base_recipe_macros

3) MACRO ADJUSTMENT (per day)
   • Sum macros for the day across all meals.
   • If totals differ from the user's daily targets, adjust the "quantity" field for meals until daily totals are within ±5% of the user's targets.
   • Respect ALL banned keywords/ingredients from context.

4) JSON MEAL PLAN (scratch-step)
   • Build JSON matching this schema (no comments):

     {{
       "meals": [
         {{
           "name": "Grilled Chicken & Rice",
           "quantity": 1.5,
           "calories": 700,
           "proteins": 45,
           "carbs": 60,
           "fats": 20,
           "sequence": 1
         }}
       ]
     }}

   • Ensure the summed macros for the day are within ±5% of the targets.
   • IMPORTANT: Every "name" MUST match a recipe name previously returned by recipe_semantic_search. 

5) TOOL CALL (FINALIZE)
   • Call return_final_answer_tool with the EXACT JSON meal plan.
"""

# ============================================================================
# PINECONE
# ============================================================================

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
recipe_index = pc.Index("syntrafit-meals-nutrition")

def extract_meal_names(data):
    data = data['result']
    return [{"id" : hit['_id'] ,"name": hit["fields"]["name"], "calories":  hit["fields"]["calories"],"carbs":  hit["fields"]["carbs"], "protein": hit["fields"]["proteins"], "fat":  hit["fields"]["fats"]}  for hit in data["hits"] if "fields" in hit and "name" in hit["fields"]]

# ============================================================================
# REWARD FUNCTIONS (V2)
# ============================================================================

# Utils
def _extract_first_json_segment(s: str) -> str | None:
    start_candidates = [s.find('{'), s.find('[')]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates: return None
    start = min(start_candidates)
    depth_obj, depth_arr, in_str, esc = 0, 0, False, False
    opener = s[start]
    want_obj, want_arr = opener == '{', opener == '['
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{': depth_obj += 1
            elif ch == '}': depth_obj -= 1
            elif ch == '[': depth_arr += 1
            elif ch == ']': depth_arr -= 1
            if want_obj and depth_obj == 0 and depth_arr == 0 and i >= start: return s[start:i+1]
            if want_arr and depth_arr == 0 and depth_obj == 0 and i >= start: return s[start:i+1]
    return None

def get_payload(obj):
    if hasattr(obj, "answer"): obj = obj.answer
    if isinstance(obj, str):
        try: obj = json.loads(obj)
        except:
             seg = _extract_first_json_segment(obj)
             if seg: 
                 try: obj = json.loads(seg)
                 except: pass
    if isinstance(obj, dict): return obj
    if isinstance(obj, list): return {"_root": obj}
    return {}

# 1. Schema Check
def verify_schema_v2(payload: dict) -> Tuple[float, Dict]:
    if not isinstance(payload, dict):
        return 0.0, {"error": "payload_not_dict"}
    
    if "meals" not in payload:
        return 0.0, {"error": "missing_meals_key"}
        
    meals = payload["meals"]
    if not isinstance(meals, list):
        return 0.0, {"error": "meals_not_list"}
        
    if len(meals) == 0:
        return 0.0, {"error": "empty_meals_list"}
        
    required_keys = ["name", "quantity", "calories", "proteins", "carbs", "fats"]
    for i, meal in enumerate(meals):
        if not isinstance(meal, dict):
            return 0.0, {"error": f"meal_{i}_not_dict"}
        for k in required_keys:
            if k not in meal:
                return 0.0, {"error": f"meal_{i}_missing_{k}"}
                
    return 1.0, {"status": "valid"}

# 2. Strict Macro Check (+/- 5%)
def verify_macros_strict(payload: dict, targets: Dict[str, float], tolerance=0.05) -> Tuple[float, Dict]:
    meals = payload.get("meals", [])
    if not meals: return 0.0, {"error": "no_meals"}
    
    total_cals = sum(float(m.get("calories", 0)) for m in meals)
    total_prot = sum(float(m.get("proteins", 0)) for m in meals)
    total_carb = sum(float(m.get("carbs", 0)) for m in meals)
    total_fat = sum(float(m.get("fats", 0)) for m in meals)
    
    errors = {}
    passed = True
    
    # Check Calories
    if targets.get("calories"):
        target = float(targets["calories"])
        if target > 0:
            err = abs(total_cals - target) / target
            errors["calories"] = err
            if err > tolerance: passed = False
            
    # Check Protein
    if targets.get("protein"):
        target = float(targets["protein"])
        if target > 0:
            err = abs(total_prot - target) / target
            errors["protein"] = err
            if err > tolerance: passed = False
            
    # Optional checks if targets exist
    if targets.get("carbs"):
        target = float(targets["carbs"])
        if target > 0:
            err = abs(total_carb - target) / target
            errors["carbs"] = err
            if err > tolerance and tolerance > 0.0: passed = False
            
    if targets.get("fat"):
        target = float(targets["fat"])
        if target > 0:
            err = abs(total_fat - target) / target
            errors["fat"] = err
            if err > tolerance and tolerance > 0.0: passed = False
            
    score = 1.0 if passed else 0.0
    return score, {"totals": {"cal": total_cals, "prot": total_prot}, "errors": errors, "passed": passed}

# 3. Variety Heuristic
def verify_variety_heuristic(payload: dict) -> Tuple[float, Dict]:
    meals = payload.get("meals", [])
    if not meals: return 0.0, {"reason": "no_meals"}
    
    # Check number of meals (aim for 3+)
    num_meals = len(meals)
    if num_meals < 3:
        return 0.0, {"reason": f"too_few_meals_{num_meals}"}
        
    # Check unique meal names (aim for 3+)
    names = [m.get("name", "").lower().strip() for m in meals]
    unique_names = set(names)
    if len(unique_names) < 3:
        return 0.0, {"reason": f"low_variety_{len(unique_names)}_unique"}
        
    return 1.0, {"unique_meals": len(unique_names), "total_meals": num_meals}

# 4. LLM Variety Judge
class VarietyJudgeResponse(BaseModel):
    score: float
    reason: str

@retry(stop=stop_after_attempt(2))
async def llm_variety_judge(scenario_text, plan_json):
    messages = [
        {"role": "system", "content": VARIETY_JUDGE_PROMPT},
        {"role": "user", "content": f"Context: {scenario_text}\n\nPlan: {json.dumps(plan_json)}"}
    ]
    try:
        response = await acompletion(
            model="openai/gpt-4o-mini", # Or use the local model if needed
            messages=messages,
            response_format=VarietyJudgeResponse
        )
        content = response.choices[0].message.content
        # If response_format handled it, we get a json string
        return json.loads(content)
    except Exception as e:
        print(f"Judge Error: {e}")
        return {"score": 0.5, "reason": "judge_error"}

# Main Reward Wrapper
async def combined_reward_v2(payload: dict, scenario_data: Scenario, traj):
    # 1. Schema
    r_schema, info_schema = verify_schema_v2(payload)
    if r_schema < 1.0:
        return 0.0, {"failure": "schema", "info": info_schema}
        
    # 2. Macros
    targets = {
        "calories": scenario_data.daily_cal_target,
        "protein": scenario_data.daily_prot_target,
        "carbs": scenario_data.daily_carb_target,
        "fat": scenario_data.daily_fat_target
    }
    r_macro, info_macro = verify_macros_strict(payload, targets, tolerance=0.05)
    
    # 3. Variety Heuristic
    r_variety_h, info_variety = verify_variety_heuristic(payload)
    
    # 4. LLM Judge (Only if basic checks pass to save cost/time)
    r_variety_llm = 0.0
    if r_macro > 0.0 and r_variety_h > 0.0:
         judge_res = await llm_variety_judge(scenario_data.question, payload)
         r_variety_llm = judge_res.get("score", 0.0)
         info_variety["llm_reason"] = judge_res.get("reason")
    else:
        # Penalize if basic variety check fails
        pass

    # Weighted Sum
    # Macro accuracy is paramount -> 0.4
    # Schema is a gate (already handled, if 0 return 0)
    # Variety Heuristic -> 0.3
    # LLM Variety -> 0.3
    
    # If macros fail, score is low
    if r_macro == 0.0:
        final_score = 0.1 # participation award
    else:
        # Base score from macros
        final_score = 0.4 
        # Add variety
        final_score += (0.3 * r_variety_h)
        # Add LLM score
        final_score += (0.3 * r_variety_llm)
        
    info = {
        "r_schema": r_schema,
        "r_macro": r_macro,
        "r_variety_h": r_variety_h,
        "r_variety_llm": r_variety_llm,
        "macro_info": info_macro,
        "variety_info": info_variety
    }
    
    return final_score, info

# ============================================================================
# TRAJECTORY & ROLLOUT
# ============================================================================

class ProjectTrajectory(art.Trajectory):
    final_answer: FinalAnswer | None = None

class FitnessScenario(BaseModel):
    step: int
    scenario: Scenario

@weave.op
async def rollout(model: art.Model, fitness_scenario: FitnessScenario) -> ProjectTrajectory:
    scenario = fitness_scenario.scenario
    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": str(scenario.id), "step": fitness_scenario.step},
    )
    
    final_answer = None

    @tool
    def recipe_semantic_search(meal_query: str, k: int = 5) -> str:
      """Search the recipe index for the most similar recipes to the query."""
      results = recipe_index.search(
          namespace="syntrafit",
          query={"top_k": k, "inputs": {'text': meal_query}}
      )
      res_list = extract_meal_names(results)
      traj.messages_and_choices.append(
          {"role": "tool_log", "content": json.dumps({"tool": "recipe_search", "query": meal_query, "result": res_list})}
      )
      return json.dumps(res_list)

    @tool
    def return_final_answer_tool(answer: str) -> dict:
        """Return the final answer (daily meal plan) in the correct format """
        nonlocal final_answer
        payload = get_payload(answer)
        final_answer = FinalAnswer(answer=payload)
        return final_answer.model_dump()

    chat_model = init_chat_model(f"{model.name}", temperature=0.2)
    react_agent = create_react_agent(chat_model, [recipe_semantic_search, return_final_answer_tool])
    
    try:
        res = await react_agent.ainvoke(
            {"messages": [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=scenario.question)]},
            config={"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 20}
        )
        
        if final_answer:
            payload = get_payload(final_answer)
            score, info = await combined_reward_v2(payload, scenario, traj)
            traj.reward = score
            traj.final_answer = final_answer
            traj.metrics.update(info)
            traj.metrics["correct"] = score
            print(f"Step {fitness_scenario.step} | ID {scenario.id} | Reward: {score:.3f}")
        else:
            traj.reward = -0.1
            traj.metrics["correct"] = 0.0
            
    except Exception as e:
        print(f"Error in rollout: {e}")
        traj.reward = -0.1
        traj.messages_and_choices.append({"role": "system", "content": f"Error: {e}"})

    return traj

# ============================================================================
# MAIN
# ============================================================================

async def main():
    global model, backend
    set_all_seeds(SEED)
    
    # Check GPU availability and verify tensor parallel size
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    num_gpus = torch.cuda.device_count()
    print(f"🔍 Detected {num_gpus} GPU(s)")
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        print(f"📌 CUDA_VISIBLE_DEVICES={cuda_visible}")
        # Verify the number matches
        visible_count = len([x for x in cuda_visible.split(",") if x.strip()])
        if visible_count != num_gpus:
            print(f"⚠️  Warning: CUDA_VISIBLE_DEVICES specifies {visible_count} device(s) but PyTorch sees {num_gpus}")
    
    # Print GPU memory info
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    # Use actual GPU count if TENSOR_PARALLEL_SIZE exceeds available GPUs
    actual_tp_size = min(TENSOR_PARALLEL_SIZE, num_gpus)
    if TENSOR_PARALLEL_SIZE > num_gpus:
        print(f"⚠️  Warning: TENSOR_PARALLEL_SIZE ({TENSOR_PARALLEL_SIZE}) exceeds available GPUs ({num_gpus}). Using {actual_tp_size} GPUs.")
    
    if actual_tp_size < TENSOR_PARALLEL_SIZE:
        print(f"📊 Using tensor_parallel_size={actual_tp_size} (requested {TENSOR_PARALLEL_SIZE})")
    
    print(f"🤖 Initializing model: {MODEL_NAME}")
    print(f"   Base model: {BASE_MODEL_NAME}")
    print(f"   Project: {PROJECT_NAME}")
    
    model = art.TrainableModel(
        name=MODEL_NAME, 
        project=PROJECT_NAME,
        base_model=BASE_MODEL_NAME
        )
    
    # Configure tensor parallelization for multi-GPU support
    print(f"⚙️  Configuring model with tensor_parallel_size={actual_tp_size}")
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=MAX_SEQ_LENGTH,
        ),
        engine_args=art.dev.EngineArgs(
            enforce_eager=ENFORCE_EAGER,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=actual_tp_size,
        ),
    )
    
    print("🔧 Initializing backend...")
    backend = LocalBackend(in_process=True, path="./.art")
    
    print("📝 Registering model with backend (this may take a moment)...")
    try:
        await model.register(backend)
        print("✅ Model registered successfully!")
    except Exception as e:
        print(f"❌ Error during model registration: {e}")
        print(f"💡 Troubleshooting tips:")
        print(f"   - Verify you have {actual_tp_size} GPU(s) available")
        print(f"   - Check GPU memory: nvidia-smi")
        print(f"   - Try reducing GPU_MEMORY_UTILIZATION from {GPU_MEMORY_UTILIZATION} to 0.6")
        print(f"   - Try reducing tensor_parallel_size to 1 for single GPU")
        raise
    
    if os.getenv("WANDB_API_KEY"):
        weave.init(model.project, settings={"print_call_link": False})

    from art.utils import iterate_dataset
    from art.langgraph import wrap_rollout

    training_iterator = iterate_dataset(
        [s for s in scenarios_list if s.split == "train"],
        groups_per_step=2,
        num_epochs=3,
        initial_step=await model.get_step()
    )
    
    for batch in training_iterator:
        print(f"Step {batch.step}")
        groups = []
        for item in batch.items:
            groups.append(art.TrajectoryGroup(
                (wrap_rollout(model, rollout)(model, FitnessScenario(step=batch.step, scenario=item.model_dump())) for _ in range(4))
            ))
            
        finished_groups = await art.gather_trajectory_groups(groups, max_exceptions=5)
        await model.train(finished_groups, config=art.TrainConfig(learning_rate=1e-5))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

