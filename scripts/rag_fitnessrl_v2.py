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

BASE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "fitness-agent-langgraph-14B-qwen2.5-003"
PROJECT_NAME = "fitness-agent-langgraph-rag-v2"
SEED = 42
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85
MAX_SEQ_LENGTH = 8192
ENFORCE_EAGER = True

# TRAINING CONFIGURATION
TRAINING_GROUPS_PER_STEP = 2
TRAINING_NUM_EPOCHS = 3
TRAINING_ROLLOUTS_PER_GROUP = 12
TRAINING_LEARNING_RATE = 1e-5
TRAINING_MAX_STEPS = 150
TRAINING_VALIDATION_EVERY = 10
TRAINING_VALIDATION_SAMPLES = 10


# Import unsloth FIRST before any other ML libraries to avoid circular import issues
# This is critical for tensor parallelism with vLLM worker spawning
import os
import sys

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
from pydantic import BaseModel, Field, field_validator, ValidationError
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

# Provenance reward (tool-macro consistency)
try:
    from src.env.provenance_reward import provenance_reward_names_only_totals_only
    print("[DEBUG] Successfully imported provenance_reward from src.env")
except Exception as e:
    print(f"[ERROR] Failed to import provenance_reward: {e}")
    provenance_reward_names_only_totals_only = None

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

# 2. Continuous Macro Check (smooth reward)
def verify_macros_continuous(payload: dict, targets: Dict[str, float]) -> Tuple[float, Dict]:
    meals = payload.get("meals", [])
    if not meals:
        return 0.0, {"error": "no_meals"}

    total_cals = sum(float(m.get("calories", 0)) for m in meals)
    total_prot = sum(float(m.get("proteins", 0)) for m in meals)
    total_carb = sum(float(m.get("carbs", 0)) for m in meals)
    total_fat = sum(float(m.get("fats", 0)) for m in meals)

    totals = {
        "calories": total_cals,
        "protein": total_prot,
        "carbs": total_carb,
        "fat": total_fat,
    }

    errors: Dict[str, float] = {}
    for key in ["calories", "protein", "carbs", "fat"]:
        target_val = targets.get(key)
        if target_val is None:
            continue
        target = float(target_val)
        if target > 0:
            errors[key] = abs(totals[key] - target) / target

    if not errors:
        return 0.0, {"error": "no_targets"}

    mean_err = sum(errors.values()) / len(errors)
    score = max(0.0, 1.0 - mean_err)
    return score, {"totals": totals, "errors": errors, "mean_error": mean_err}

# 3. Variety Heuristic (smooth reward)
def verify_variety_heuristic(payload: dict) -> Tuple[float, Dict]:
    meals = payload.get("meals", [])
    if not meals: return 0.0, {"reason": "no_meals"}

    num_meals = len(meals)
    names = [m.get("name", "").lower().strip() for m in meals if m.get("name")]
    unique_names = set(names)

    meals_score = min(1.0, num_meals / 3.0)
    variety_score = min(1.0, len(unique_names) / 3.0)
    score = 0.5 * meals_score + 0.5 * variety_score

    reason = []
    if num_meals < 3:
        reason.append(f"too_few_meals_{num_meals}")
    if len(unique_names) < 3:
        reason.append(f"low_variety_{len(unique_names)}_unique")

    return score, {"unique_meals": len(unique_names), "total_meals": num_meals, "reason": reason or ["ok"]}

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
            model="openai/gpt-4o-mini",
            messages=messages,
            response_format=VarietyJudgeResponse
        )
        content = response.choices[0].message.content
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
    r_macro, info_macro = verify_macros_continuous(payload, targets)

    # 3. Variety Heuristic
    r_variety_h, info_variety = verify_variety_heuristic(payload)

    # 4. LLM Judge (Only if schema is valid to save cost/time)
    r_variety_llm = 0.0
    if r_variety_h > 0.0:
         judge_res = await llm_variety_judge(scenario_data.question, payload)
         r_variety_llm = judge_res.get("score", 0.0)
         info_variety["llm_reason"] = judge_res.get("reason")

    # 5. Provenance (tool-macro consistency)  ✅ MITIGATION #3: contain exceptions
    if provenance_reward_names_only_totals_only is None:
        r_prov, info_prov = 0.0, {"reason": "provenance_reward_unavailable"}
    else:
        try:
            r_prov, info_prov = provenance_reward_names_only_totals_only(payload, traj)
        except Exception as e:
            r_prov, info_prov = 0.0, {"reason": f"provenance_exception: {type(e).__name__}: {e}"}

    final_score = (
        (0.60 * r_macro)
        + (0.20 * r_variety_h)
        # + (0.15 * r_variety_llm)
        + (0.20 * r_prov)
    )

    info = {
        "r_schema": r_schema,
        "r_macro": r_macro,
        "r_variety_h": r_variety_h,
        "r_variety_llm": r_variety_llm,
        "r_provenance": r_prov,
        "macro_info": info_macro,
        "variety_info": info_variety,
        "provenance_info": info_prov,
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

    final_answer: FinalAnswer | None = None

    @tool
    def recipe_semantic_search(meal_query: str, k: int = 5) -> str:
      """Search the recipe index for the most similar recipes to the query."""
      results = recipe_index.search(
          namespace="syntrafit",
          query={"top_k": k, "inputs": {'text': meal_query}}
      )
      res_list = extract_meal_names(results)
      traj.messages_and_choices.append(
          {
              "role": "tool_log",
              "content": json.dumps(
                  {
                      "tool": "recipe_search",
                      "query": meal_query,
                      "result": res_list,
                      "end": {"tool": "recipe_semantic_search", "result": res_list},
                  }
              ),
          }
      )
      return json.dumps(res_list)

    # ✅ MITIGATION #1: accept dict, not str (dramatically fewer tool-call failures)
    @tool
    def return_final_answer_tool(answer: dict) -> dict:
        """Return the final answer (daily meal plan) in the correct format."""
        nonlocal final_answer
        payload = get_payload(answer)  # defensive, but should already be dict
        final_answer = FinalAnswer(answer=payload)  # pydantic validation here
        return final_answer.model_dump()

    chat_model = init_chat_model(f"{model.name}", temperature=0.2)
    react_agent = create_react_agent(chat_model, [recipe_semantic_search, return_final_answer_tool])

    # ✅ MITIGATION #2: one-shot "repair" attempt if no final tool call happened
    async def _attempt_repair_if_missing_final():
        nonlocal final_answer
        if final_answer is not None:
            return

        traj.messages_and_choices.append({"role": "system", "content": "[NO_FINAL_TOOL_CALL] attempting repair invoke"})

        repair_prompt = """
You MUST now output ONLY a single tool call to return_final_answer_tool.
Do NOT write any normal text.
The tool argument must be a JSON OBJECT (a dict) with schema: {"meals":[{...}]}.
"""

        try:
            await react_agent.ainvoke(
                {"messages": [
                    SystemMessage(content=PLANNER_PROMPT),
                    HumanMessage(content=scenario.question),
                    SystemMessage(content=repair_prompt),
                ]},
                config={"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 20}
            )
        except Exception as e:
            traj.messages_and_choices.append({"role": "system", "content": f"[REPAIR_EXCEPTION] {type(e).__name__}: {e}"})

    try:
        max_retries = 3
        res = None

        for attempt in range(max_retries):
            try:
                res = await react_agent.ainvoke(
                    {"messages": [SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=scenario.question)]},
                    config={"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 20}
                )
                break
            except ValidationError as ve:
                if attempt < max_retries - 1:
                    print(f"⚠️ Validation error in rollout (attempt {attempt+1}/{max_retries}): {ve}. Retrying...")
                    continue
                else:
                    raise ve
            except Exception as e:
                err_str = str(e)
                if "Input should be a valid dictionary" in err_str and attempt < max_retries - 1:
                     print(f"⚠️ Tool call parsing error (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                     continue
                raise e

        # ✅ MITIGATION #2: repair if model finished without calling final tool
        if final_answer is None:
            await _attempt_repair_if_missing_final()

        if final_answer:
            payload = get_payload(final_answer)
            score, info = await combined_reward_v2(payload, scenario, traj)
            traj.reward = score
            traj.final_answer = final_answer

            # Only store numeric metrics (ART can't handle dicts)
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    traj.metrics[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            traj.metrics[f"{key}_{sub_key}"] = sub_value

            traj.metrics["correct"] = score
            print(f"Step {fitness_scenario.step} | ID {scenario.id} | Reward: {score:.3f}")
        else:
            traj.reward = -0.1
            traj.metrics["correct"] = 0.0
            traj.messages_and_choices.append({"role": "system", "content": "[NO_FINAL_TOOL_CALL] giving -0.1"})

    except Exception as e:
        print(f"Error in rollout: {e}")
        traj.reward = 0.0
        traj.messages_and_choices.append({"role": "system", "content": f"[EXCEPTION] {type(e).__name__}: {e}"})

    return traj

# ============================================================================
# MAIN
# ============================================================================

async def main():
    global model, backend
    set_all_seeds(SEED)

    print(f"🤖 Initializing model: {MODEL_NAME}")
    print(f"   Base model: {BASE_MODEL_NAME}")
    print(f"   Project: {PROJECT_NAME}")

    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL_NAME
    )

    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=MAX_SEQ_LENGTH,
        ),
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        ),
    )

    print("🔧 Initializing backend...")
    backend = LocalBackend(in_process=True, path="./.art")

    print("📝 Registering model with backend...")
    await model.register(backend)
    print("✅ Model registered successfully!")

    if os.getenv("WANDB_API_KEY"):
        weave.init(model.project, settings={"print_call_link": False})

    from art.utils import iterate_dataset
    from art.langgraph import wrap_rollout

    # Split scenarios into train and validation
    train_scenarios = [s for s in scenarios_list if s.split == "train"]
    val_scenarios = [s for s in scenarios_list if s.split == "test"]

    print(f"📊 Training scenarios: {len(train_scenarios)}")
    print(f"📊 Validation scenarios: {len(val_scenarios)}")

    # =========================================================================
    # VALIDATION FUNCTION
    # =========================================================================
    async def run_validation(step: int):
        """Run validation on held-out test scenarios."""
        print(f"\n🔍 Running validation at step {step}...")

        val_sample = val_scenarios[:TRAINING_VALIDATION_SAMPLES] if len(val_scenarios) > TRAINING_VALIDATION_SAMPLES else val_scenarios

        val_groups = []
        for scenario in val_sample:
            val_groups.append(art.TrajectoryGroup(
                [wrap_rollout(model, rollout)(model, FitnessScenario(step=step, scenario=scenario))]
            ))

        finished_val_groups = await art.gather_trajectory_groups(
            val_groups,
            pbar_desc="validation",
            max_exceptions=len(val_sample)
        )

        total_reward = 0.0
        total_correct = 0.0
        count = 0

        for group in finished_val_groups:
            for traj in group:
                total_reward += traj.reward
                total_correct += traj.metrics.get("correct", 0.0)
                count += 1

        avg_reward = total_reward / count if count > 0 else 0.0
        avg_correct = total_correct / count if count > 0 else 0.0

        print(f"📈 Validation Results (step {step}):")
        print(f"   Avg Reward: {avg_reward:.4f}")
        print(f"   Avg Correct: {avg_correct:.4f}")
        print(f"   Samples: {count}")

        if os.getenv("WANDB_API_KEY"):
            try:
                import wandb
                wandb.log({
                    "val/avg_reward": avg_reward,
                    "val/avg_correct": avg_correct,
                    "val/num_samples": count,
                    "step": step,
                })
            except Exception as e:
                print(f"⚠️  W&B logging error: {e}")

        return {"avg_reward": avg_reward, "avg_correct": avg_correct, "count": count}

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    training_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=TRAINING_GROUPS_PER_STEP,
        num_epochs=TRAINING_NUM_EPOCHS,
        initial_step=await model.get_step()
    )

    for batch in training_iterator:
        current_step = batch.step
        print(f"\n{'='*60}")
        print(f"Step {current_step} | Epoch {batch.epoch}")
        print(f"{'='*60}")

        if current_step >= TRAINING_MAX_STEPS:
            print(f"🛑 Reached max_steps ({TRAINING_MAX_STEPS}), stopping training.")
            break

        groups = []
        for item in batch.items:
            groups.append(art.TrajectoryGroup(
                (wrap_rollout(model, rollout)(model, FitnessScenario(step=current_step, scenario=item))
                 for _ in range(TRAINING_ROLLOUTS_PER_GROUP))
            ))

        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="training",
            max_exceptions=TRAINING_ROLLOUTS_PER_GROUP * len(batch.items)
        )

        train_rewards = []
        for group in finished_groups:
            for traj in group:
                train_rewards.append(traj.reward)

        if train_rewards:
            avg_train_reward = sum(train_rewards) / len(train_rewards)
            print(f"📊 Training batch avg reward: {avg_train_reward:.4f}")

        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=TRAINING_LEARNING_RATE)
        )
        print(f"✅ Training step {current_step} complete")

        if current_step > 0 and current_step % TRAINING_VALIDATION_EVERY == 0:
            _ = await run_validation(current_step)

    print("\n" + "="*60)
    print("🏁 Training complete! Running final validation...")
    print("="*60)
    final_val = await run_validation(current_step)

    print("\n🎉 Training finished!")
    print(f"   Final validation reward: {final_val['avg_reward']:.4f}")
    print(f"   Final validation correct: {final_val['avg_correct']:.4f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())