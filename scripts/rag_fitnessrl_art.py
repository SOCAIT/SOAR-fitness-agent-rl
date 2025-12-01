#!/usr/bin/env python3
"""
RAG-FitnessRL-Art Training Script

Fitness Agent training using ART with LangGraph and RAG.
This script trains a model to generate nutrition plans using reinforcement learning
with provenance-based rewards.

PREREQUISITES:
--------------
1. Install dependencies: `uv sync`
2. Set up environment variables in .env file:
   - OPENAI_API_KEY (required)
   - PINECONE_API_KEY (required)
   - WANDB_API_KEY (optional, for logging)

USAGE:
------
On a remote VM with GPU:
    python scripts/rag_fitnessrl_art.py

The script will:
1. Load training scenarios from data/fitness_scenarios.jsonl
2. Initialize the Qwen 2.5 7B model
3. Set up RAG with Pinecone for recipe search
4. Run the training loop with ART
5. Save checkpoints in .art/ directory

REQUIREMENTS:
-------------
- GPU with at least 16GB VRAM (recommended)
- Python 3.10+
- All dependencies from pyproject.toml
"""

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
            # Unwrap nested FinalAnswer by mistake
            if isinstance(v, FinalAnswer):
                return v.answer
            # Parse JSON string
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError as e:
                    raise ValueError(f"answer must be a JSON object string or dict; got invalid JSON: {e}")
            # Already a dict
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

# Load the dataset from the JSONL file
dataset_path = project_root / "data" / "fitness_scenarios.jsonl"
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

dataset = load_dataset("json", data_files=str(dataset_path))

# Assuming the dataset has a "train" split, you can access it like this:
training_scenarios = dataset["train"]

print("Dataset loaded successfully!")
print(training_scenarios)

training_scenarios[-1]


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

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
    daily_cal_target = example['context']['daily_cal_target']
    daily_prot_target = example['context']['daily_prot_target']
    daily_carb_target = example['context']['daily_carb_target']
    daily_fat_target = example['context']['daily_fat_target']
    banned_keywords = example['context']['banned_keywords']

    example['daily_cal_target'] = daily_cal_target
    example['daily_prot_target'] = daily_prot_target
    example['daily_carb_target'] = daily_carb_target
    example['daily_fat_target'] = daily_fat_target
    example['banned_keywords'] = banned_keywords
    return example

def extract_context_columns(example):
    context = example['context']
    example['age'] = context.get('age')
    example['sex'] = context.get('sex')
    example['height_cm'] = context.get('height_cm')
    example['weight_kg'] = context.get('weight_kg')
    example['goal'] = context.get('goal')
    example['activity'] = context.get('activity')
    example['dietary_prefs'] = context.get('dietary_prefs')
    example['equipment'] = context.get('equipment')
    example['experience'] = context.get('experience')
    return example

def make_scenario(example):
    scenario = Scenario(
        question=example['question'],
        split=example['split'],
        id=str(example['id']),
        daily_cal_target=example['daily_cal_target'],
        daily_prot_target=example['daily_prot_target'],
        daily_carb_target=example['daily_carb_target'],
        daily_fat_target=example['daily_fat_target'],
        #banned_keywords=example['banned_keywords']
    )
    return scenario
training_scenarios = training_scenarios.map(one_day_meal_question)
training_scenarios = training_scenarios.map(combine_question_and_context)
training_scenarios = training_scenarios.map(convert_val_to_test)
training_scenarios = training_scenarios.map(get_target_nutrition_data)
training_scenarios = training_scenarios.map(extract_context_columns)
# training_scenarios = training_scenarios.map(make_scenario)
print(training_scenarios[0]['question'])

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

from collections import Counter

print(Counter(training_scenarios['split']))

scenarios_list[0]


# ============================================================================
# MODEL AND BACKEND SETUP (will be done in main())
# ============================================================================

# Model configuration (global variables for use in rollout function)
BASE_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_NAME = "fitness-agent-langgraph-14B-qwen2.5-002"
PROJECT_NAME = "fitness-agent-langgraph-rag"

# These will be initialized in main()
model = None
backend = None


# ============================================================================
# PROMPTS AND CONFIGURATION
# ============================================================================

RULER_JUDGE_PROMPT = """
You are a strict evaluator for weekly nutrition/workout plans for SyntraFit.

Score each ASSISTANT response on 0–1 (decimals allowed) based on:

[Critical]
1) STRUCTURE: Valid JSON structure per task .
2) TARGET FIT: Daily calories & protein within ±5% of user's target_nutrition_data (if nutrition task).
3) DIETARY RULES: No banned items present (e.g., “egg”, “shellfish”) if user disallows.
4) TOOLING LOGIC: The content is plausibly the result of correct tools (recipe_semantic_search for recipes; get_available_exercises for exercises).

[Quality]
6) REALISM: Meals/exercises are realistic, balanced, and varied across the week; sequencing makes sense (snacks vs meals; rest days).
7) CLARITY: Names are specific (e.g., “Grilled Salmon & Quinoa” not “healthy bowl”).

Return ONLY a float in [0,1]. Heavily penalize if structure or targets are wrong.
"""

MAX_TURNS = 30

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
- You may take up to {MAX_TURNS} turns to find the answer.

============================================================
NUTRITION PLAN PIPELINE
============================================================

1) PLAN SKELETON
   • Generate a one-day meal plan for the user. The plan should have meals that fulfill the user's daily macro targets.
   • Create a reasonable number of meals (meals can include snacks). Base the count/portions on the user's macro targets.
   • Meal names in the final plan MUST be recipe names returned by recipe_semantic_search (do not invent names).

2) USE RECIPE SEMANTIC SEARCH TOOL
   • Use recipe_semantic_search to find relevant recipes with correct nutrition info.
   • You MUST ALWAYS use the macros retrieved from the tool and NOT infer your own data.
   • For each meal you include:
     - Set "name" to the exact recipe name from the tool
     - Set "quantity" to a multiplier (e.g., 1.0 for one serving, 1.5 for 1.5 servings, 0.5 for half serving)
     - Calculate macros as: quantity × base_recipe_macros
     - Example: If recipe has 400 cal and quantity=1.5, then calories should be 600
   • If a candidate recipe includes any banned keywords/ingredients from context, discard it and search again.

3) MACRO ADJUSTMENT (per day)
   • Sum macros for the day across all meals.
   • If totals differ from the user's daily targets, adjust the "quantity" field for meals (still using tool macros) until daily totals are within ±5% of the user's targets (calories/protein/carbs/fat).
   • You can use fractional quantities like 0.75, 1.25, 1.5 to hit precise macro targets.
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

   • The "quantity" field is a multiplier (can be fractional like 1.5, 0.75, 2.0) that represents portions/servings.
   • Macros (calories, proteins, carbs, fats) should be: quantity × base_recipe_macros from recipe_semantic_search.
   • Ensure the summed macros for the day are within ±5% of the targets.
   • IMPORTANT: Every "name" MUST match a recipe name previously returned by recipe_semantic_search. 
   • The quantity field allows precise macro targeting by scaling recipe portions.

5) IF YOU REACHED MAX_TURNS and you have not found a final answer, return the best final answer you can with all information you gathered.

6) TOOL CALL (FINALIZE)
   • Call return_final_answer_tool with:
     - answer = the EXACT JSON meal plan (stringified if needed)

"""


# ============================================================================
# PINECONE SETUP
# ============================================================================

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

recipe_index_name = "syntrafit-meals-nutrition"
exercise_index_name = "syntrafit-exercises"

recipe_index = pc.Index(recipe_index_name)
exercise_index = pc.Index(exercise_index_name)

def extract_meal_names(data):
    data = data['result']
    return_data = []


    return [{"id" : hit['_id'] ,"name": hit["fields"]["name"], "calories":  hit["fields"]["calories"],"carbs":  hit["fields"]["carbs"], "protein": hit["fields"]["proteins"], "fat":  hit["fields"]["fats"]}  for hit in data["hits"] if "fields" in hit and "name" in hit["fields"]]

 # Search the dense index
results = recipe_index.search(
          namespace="syntrafit",
          query={
              "top_k": 2,
              "inputs": {
                  'text': " Chicken and rice healthy"
              }
          },
          rerank={
          "model": "bge-reranker-v2-m3",
          "top_n": 2,
          "rank_fields": [ "name"]
    },
      )

print(results)

extract_meal_names(results)


# ============================================================================
# WEAVE INITIALIZATION (will be done in main() after model setup)
# ============================================================================


# ============================================================================
# JUDGE AND EVALUATION
# ============================================================================

class NutritionJudgeResponse(BaseModel):
    reasoning: str = Field(description="Explanation of the reasoning process.")
    score: float = Field(description="Score between 0 and 1.")


@retry(stop=stop_after_attempt(3))
async def nutrition_judge_score(scenario, answer):


    # # 2) RULER judge
    # judged = await ruler_score_group(
    #     group, "openai/o4-mini",
    #     system_instruction=RULER_JUDGE_PROMPT, debug=False
    # )

    messages = [
        {"role": "system", "content": RULER_JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {scenario.question}\n"

                f"Assistant Answer: {answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    response = await acompletion(
        model="openai/gpt-4.1",
        messages=messages,
        response_format=NutritionJudgeResponse,
    )

    # Access attributes directly instead of using .get()
    return NutritionJudgeResponse(reasoning=response.choices[0].message.content.reasoning, score=response.choices[0].message.content.score)


# ============================================================================
# PAYLOAD PARSING UTILITIES
# ============================================================================

def _extract_first_json_segment(s: str) -> str | None:
    """Extract the first complete top-level JSON object or array from a noisy string."""
    start_candidates = [s.find('{'), s.find('[')]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)

    # Track either {} or [] depth; handle strings and escapes
    depth_obj = 0
    depth_arr = 0
    in_str = False
    esc = False

    opener = s[start]
    want_obj = opener == '{'
    want_arr = opener == '['

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

            # Completed top-level
            if want_obj and depth_obj == 0 and depth_arr == 0 and i >= start:
                return s[start:i+1]
            if want_arr and depth_arr == 0 and depth_obj == 0 and i >= start:
                return s[start:i+1]
    return None  # unbalanced / truncated

def _loads_loose(s: str):
    """Try multiple strategies to convert a string into JSON (object or array)."""
    # 1) Try direct
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        v = None

    # 2) If failed OR result is a string (double-encoded), try to decode up to 3 times
    tries = 0
    while isinstance(v, str) and tries < 3:
        tries += 1
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            break

    if v is not None and not isinstance(v, str):
        return v

    # 3) Try to extract first clean JSON segment from noisy logs and parse it
    seg = _extract_first_json_segment(s)
    if seg is not None:
        try:
            return json.loads(seg)
        except json.JSONDecodeError:
            pass

    # Give up
    raise json.JSONDecodeError("Could not parse JSON from string", s, 0)

def get_payload(obj):
    """Return a dict payload from FinalAnswer/str/dict.
    - Accepts dicts and top-level arrays (wrapped under '_root')
    - Unwraps .answer
    - Parses double-encoded JSON strings
    - Extracts JSON from noisy/log-polluted strings
    - Supports Pydantic and dataclasses
    """
    # unwrap FinalAnswer wrapper
    if hasattr(obj, "answer"):
        obj = obj.answer

    # decode bytes
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8", errors="replace")

    # parse JSON string (robust)
    if isinstance(obj, str):
        try:
            obj = _loads_loose(obj)
        except json.JSONDecodeError:
            return {"_error": "invalid_json_string", "_raw": obj}

    # pydantic compatibility
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "dict"):
        try:
            obj = obj.dict()
        except TypeError:
            # some dataclasses also have .dict attribute conflicts; handle below
            pass

    # dataclasses
    if is_dataclass(obj):
        obj = asdict(obj)

    # final shape handling
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):  # allow top-level arrays while keeping a dict payload
        return {"_root": obj}

    return {"_error": f"unexpected_type:{type(obj).__name__}", "_raw": str(obj)}


# ============================================================================
# PROVENANCE REWARD FUNCTION
# ============================================================================
# Matches plan meals to recipe_semantic_search results by NAME only
# and verifies totals are a consistent multiple of tool macros.

# ---------- utils ----------
def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # collapse spaces, remove trivial punctuation (& , .  multiple spaces)
    s = s.lower()
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)           # keep only letters/digits/spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _rel_err(true, pred):
    if true == 0:
        return 0.0 if abs(pred) < 1e-6 else 1.0
    return abs(pred - true) / max(1.0, abs(true))

def _infer_multiplier(unit: dict, totals: dict, min_dims=2):
    """
    Infer multiplier m from totals using median of ratios across available macros.
    Requires at least `min_dims` macros present. Returns (m, worst_rel_err) or (None, None).
    """
    ratios, dims = [], []
    for k in ("calories", "carbs", "protein", "fat"):
        t = totals.get(k)
        u = unit.get(k)
        if t is None or u in (None, 0):
            continue
        try:
            ratios.append(float(t) / float(u))
            dims.append(k)
        except Exception:
            pass
    if len(ratios) < min_dims:
        return None, None
    m = median(ratios)
    if not (m > 0 and math.isfinite(m)):
        return None, None
    worst = 0.0
    for k in dims:
        pred = unit[k] * m
        err = _rel_err(pred, float(totals[k]))
        if err > worst:
            worst = err
    return m, worst

def _flatten_plan_meals(payload: dict):
    """Expect schema:
    {
      "meals": [
        {"name": "...", "calories": ..., "proteins": ..., "carbs": ..., "fats": ..., "sequence": ...},
        ...
      ]
    }
    """
    meals = []
    if isinstance(payload, dict) and isinstance(payload.get("meals"), list):
        for m in payload["meals"]:
            if isinstance(m, dict):
                meals.append(m)
    return meals

def _extract_totals_from_meal(meal: dict):
    """Map plan keys -> canonical keys (proteins->protein, fats->fat).
    Returns (totals_dict, explicit_quantity) where quantity is the multiplier if present.
    """
    totals = {}
    quantity = None
    
    if not isinstance(meal, dict):
        return totals, quantity
    
    # Extract explicit quantity/multiplier if present
    if "quantity" in meal:
        try:
            quantity = float(meal["quantity"])
        except (ValueError, TypeError):
            quantity = None
    
    # Extract macros
    if "calories" in meal: totals["calories"] = meal["calories"]
    if "carbs"    in meal: totals["carbs"]    = meal["carbs"]
    if "proteins" in meal: totals["protein"]  = meal["proteins"]
    if "fats"     in meal: totals["fat"]      = meal["fats"]
    
    return totals, quantity

# ---------- catalog from tool logs ----------
def _collect_catalog_from_logs_by_name(traj, tool_name="recipe_semantic_search"):
    """
    Build {normalized_name: {"raw_name": str, "macros": {calories,carbs,protein,fat}, "id": <id>}}
    from tool logs where result looks like:
      [
        {"id": "...", "name": "...", "calories": 382.0, "carbs": 35.0, "protein": 33.0, "fat": 12.0},
        ...
      ]
    or {"recipes": [ ...same list... ]}
    """
    catalog = {}
    for m in getattr(traj, "messages_and_choices", []):
        if m.get("role") != "tool_log":
            continue
        try:
            log = json.loads(m["content"])
        except Exception:
            continue
        end = log.get("end")
        if not end or end.get("tool") != tool_name:
            continue

        res = end.get("result")
        # result might be a JSON string or a Python list/dict
        if isinstance(res, str):
            try:
                res = json.loads(res)
            except Exception:
                pass

        recipes = res.get("recipes") if isinstance(res, dict) else res
        if not isinstance(recipes, list):
            continue

        for r in recipes:
            name = r.get("name")
            if not name:
                continue
            key = _norm_name(name)
            try:
                catalog[key] = {
                    "raw_name": name,
                    "id": r.get("id"),
                    "macros": {
                        "calories": float(r["calories"]),
                        "carbs": float(r["carbs"]),
                        "protein": float(r["protein"]),
                        "fat": float(r["fat"]),
                    },
                }
            except Exception:
                # skip incomplete/bad rows
                continue
    return catalog

# ---------- main reward ----------
def provenance_reward_names_only_totals_only(
    payload: dict,
    traj,
    tol_pct: float = 0.05,
    name_match_cutoff: float = 0.92,
    tool_name: str = "recipe_semantic_search",
):
    """
    Returns (score_in_[0,1], info_dict).

    Pass criteria per meal:
      1) Meal name must match (exact or fuzzy) a recipe name returned by the tool
         during this trajectory.
      2) Meal's totals (calories, carbs, proteins, fats) must be a consistent
         multiple of the tool's macros (worst relative error <= tol_pct).

    No 'servings', no per-serving fields, names-only linkage.
    """
    catalog = _collect_catalog_from_logs_by_name(traj, tool_name=tool_name)
    if not catalog:
        return 0.0, {"reason": "no_recipe_tool_usage_detected"}

    meals = _flatten_plan_meals(payload)
    if not meals:
        return 0.0, {"reason": "empty_plan"}

    checked = passed = 0
    details = []

    # precompute name list for fuzzy matching
    cat_names = list(catalog.keys())

    for meal in meals:
        checked += 1
        plan_name_raw = meal.get("name", "")
        plan_key = _norm_name(plan_name_raw)

        # Try exact normalized match first
        match_key = plan_key if plan_key in catalog else None

        # Fallback: fuzzy match on normalized names
        if match_key is None and cat_names:
            best = difflib.get_close_matches(plan_key, cat_names, n=1, cutoff=name_match_cutoff)
            if best:
                match_key = best[0]

        if match_key is None:
            details.append({
                "ok": False,
                "plan_name": plan_name_raw,
                "reason": "no_name_match_in_tool_results",
            })
            continue

        canon = catalog[match_key]["macros"]
        totals, explicit_quantity = _extract_totals_from_meal(meal)
        if not totals:
            details.append({
                "ok": False,
                "plan_name": plan_name_raw,
                "matched_recipe_name": catalog[match_key]["raw_name"],
                "reason": "no_totals_in_meal",
            })
            continue

        # If explicit quantity provided, verify macros match quantity × base_macros
        if explicit_quantity is not None:
            # Check if provided macros are close to explicit_quantity × canon
            worst_err = 0.0
            for k in ("calories", "carbs", "protein", "fat"):
                if k not in totals or k not in canon:
                    continue
                expected = canon[k] * explicit_quantity
                actual = float(totals[k])
                err = _rel_err(expected, actual)
                if err > worst_err:
                    worst_err = err
            
            ok = worst_err <= tol_pct
            m = explicit_quantity
        else:
            # Fallback: infer multiplier from totals
            m, worst_err = _infer_multiplier(canon, totals, min_dims=2)
            ok = (m is not None and worst_err is not None and worst_err <= tol_pct)

        details.append({
            "ok": ok,
            "plan_name": plan_name_raw,
            "matched_recipe_name": catalog[match_key]["raw_name"],
            "matched_recipe_id": catalog[match_key]["id"],
            "quantity": m,
            "explicit_quantity": explicit_quantity is not None,
            "worst_rel_err": worst_err,
        })

        if ok:
            passed += 1

    score = passed / max(1, checked)
    return score, {"checked": checked, "passed": passed, "items": details}


# ============================================================================
# LOCAL PROJECT IMPORTS
# ============================================================================

from src.helpers import get_exercise_list_for_prompt
from src.env.verifiers_utils import verify_nutrition_plan, verify_workout_plan, verify_nutrition_schema, verify_meal_plan_schema, verify_daily_meal_plan_macros, is_valid_json
from src.data_utils.create_synthetic_data import data
from src.env.verifiable_rewards.nutrition_rewards import nutrition_reward


# ============================================================================
# REWARD COMBINATION FUNCTIONS
# ============================================================================

def combine_reward_grpo(
    nutri: float,                 # macros/target score (any scale; we'll clamp)
    prov: float,                  # provenance score in [0,1]
    step: int | None = None,      # training step for scheduling (optional)
    gamma: float = 1.0,           # emphasis on macros
    beta: float | None = None,    # emphasis on provenance (if None -> schedule)
    eps: float = 0.03,            # floor to avoid zero-collapse in the gate
    mix: float = 0.15,            # how much dense linear part to mix in (0..0.3 works well)
    jitter: float = 1e-3,         # tiny tie-breaker within a group
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (score_for_grpo, info). Score is in [0,1].
    Designed for GRPO: adds a small dense component and jitter to preserve ranking/spread.
    """

    # ---- simple schedule for beta (increase provenance pressure over time) ----
    # def _beta_schedule(s: int | None) -> float:
    #     if s is None:  return 1.5
    #     if s < 100:    return 1.2
    #     if s < 300:    return 1.5
    #     return 1.8
    def _beta_schedule(s: int | None) -> float:
        """
        Schedule for provenance emphasis:
        - very low early (0–50): focus on macro/schema correctness
        - moderate mid (50–150): start caring about provenance
        - strong later (150+): provenance matters a lot
        """
        if s is None:
            return 1.0
        if s < 50:
            return 0.5      # very soft provenance early on
        if s < 150:
            return 1.0      # balanced
        if s < 300:
            return 1.5      # stronger provenance
        return 1.8

    B = beta if beta is not None else _beta_schedule(step)

    # ---- clamp and apply epsilon floor for the gate ----
    n = max(0.0, min(1.0, float(nutri)))
    p = max(0.0, min(1.0, float(prov)))
    n_gate = max(eps, n)
    p_gate = max(eps, p)

    # ---- sparse gate (soft-AND) ----
    gate = (n_gate ** gamma) * (p_gate ** B)
    # normalize gate to [0,1] given eps floor
    gate_min = (eps ** (gamma + B))
    if gate_min < 1.0:
        gate = (gate - gate_min) / (1.0 - gate_min)
    gate = max(0.0, min(1.0, gate))

    # ---- dense part to keep intra-group spread (rank signal even when gate fails) ----
    # small linear mix that still prefers provenance
    dense = 0.6 * p + 0.4 * n    # both already in [0,1]

    # ---- final blend + jitter (rank only; GRPO will mean-center anyway) ----
    score = (1.0 - mix) * gate + mix * dense + random.random() * jitter
    score = max(0.0, min(1.0, score))

    info = {
        "nutri_clamped": n, "prov_clamped": p,
        "gamma": gamma, "beta": B, "eps": eps,
        "mix": mix, "jitter": jitter,
        "gate": gate, "dense": dense, "score": score
    }
    return score, info


# ============================================================================
# TRAJECTORY AND ROLLOUT SETUP
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
        metadata={
            "scenario_id": str(scenario.id),
            "step": fitness_scenario.step,
        },
    )

    system_prompt = dedent(
        PLANNER_PROMPT
    )

    # Store final answer in trajectory
    final_answer = None

    # Target Nutrition
    daily_cal_target = scenario.daily_cal_target
    daily_prot_target = scenario.daily_prot_target
    daily_carb_target = scenario.daily_carb_target
    daily_fat_target = scenario.daily_fat_target


    # def log_tool(tool_name):
    #     def decorator(fn):
    #         @wraps(fn)
    #         def wrapper(*args, **kwargs):
    #             call = {
    #                 "tool": tool_name,
    #                 "ts": datetime.utcnow().isoformat(),
    #                 "args": args, "kwargs": kwargs
    #             }
    #             print(f"[TOOL START] {tool_name} args={kwargs}")
    #             traj.messages_and_choices.append({"role": "tool_log", "content": json.dumps({"start": call})})
    #             try:
    #                 out = fn(*args, **kwargs)
    #                 print(f"[TOOL END] {tool_name} result_preview={str(out)[:400]}")
    #                 traj.messages_and_choices.append({"role": "tool_log", "content": json.dumps({"end": {**call, "result": out}})})
    #                 return out
    #             except Exception as e:
    #                 print(f"[TOOL ERROR] {tool_name}: {e}")
    #                 traj.messages_and_choices.append({"role": "tool_log", "content": json.dumps({"error": {**call, "error": str(e)}})})
    #                 raise
    #         return wrapper
    #     return decorator


    def log_tool(tool_name):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                GREEN = "\033[92m"
                RESET = "\033[0m"

                call = {
                    "tool": tool_name,
                    "ts": datetime.utcnow().isoformat(),
                    "args": args,
                    "kwargs": kwargs
                }

                color_prefix = GREEN if "final_answer" in tool_name else ""
                color_reset = RESET if "final_answer" in tool_name else ""

                print(f"{color_prefix}[TOOL START] {tool_name} args={kwargs}{color_reset}")
                traj.messages_and_choices.append(
                    {"role": "tool_log", "content": json.dumps({"start": call})}
                )

                try:
                    out = fn(*args, **kwargs)
                    print(f"{color_prefix}[TOOL END] {tool_name} result_preview={str(out)[:400]}{color_reset}")
                    traj.messages_and_choices.append(
                        {"role": "tool_log", "content": json.dumps({"end": {**call, 'result': out}})}
                    )
                    return out
                except Exception as e:
                    print(f"\033[91m[TOOL ERROR] {tool_name}: {e}\033[0m")
                    traj.messages_and_choices.append(
                        {"role": "tool_log", "content": json.dumps({"error": {**call, "error": str(e)}})}
                    )
                    raise
            return wrapper
        return decorator

    @tool
    @log_tool("recipe_semantic_search")
    def recipe_semantic_search(meal_query: str, k: int = 5) -> str:
      """Search the recipe index for the most similar recipes to the query."""
      # Search the dense index
      results = recipe_index.search(
          namespace="syntrafit",
          query={
              "top_k": k,
              "inputs": {
                  'text': meal_query
              }
          }
      )

      results = extract_meal_names(results)

      print(results)

      return results


    @tool
    @log_tool("return_final_answer_tool")
    def return_final_answer_tool(answer: str) -> dict:
        """Return the final answer (daily meal plan) in the correct format """
        nonlocal final_answer
        payload = get_payload(answer)          # <-- normalize here
        final_answer = FinalAnswer(answer=payload)
        return final_answer.model_dump()


    # Create LangGraph tools
    tools = [ recipe_semantic_search,  return_final_answer_tool] #recipe_semantic_search,

    # Pass the local path to the model
    chat_model = init_chat_model(f"{model.name}", temperature=0.5)


    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, tools)
    print("LangGraph agent created!")
    print(react_agent)

    try:
        # Run the agent
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": MAX_TURNS,
        }

        print("Human Question:", scenario.question )



        # Run the agent to get the final result
        res =await react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config=config,
        )


        print("rollout_2")
        print(res)

        # Check if we got a final answer
        if final_answer:
            print("Got final answer!")
            print(final_answer)

            payload = get_payload(final_answer)    # <-- normalize again defensively
            print(payload)                         # should print a dict, not a string

            # Calculate the total reward
            total_reward = 0.0
            traj.final_answer = final_answer
            print(final_answer.answer)

            banned_keywords = scenario.banned_keywords
            print(banned_keywords)


            reward, nutri_info = nutrition_reward(
                      payload,
                      daily_cal_target=daily_cal_target,
                      daily_prot_target=daily_prot_target,
                      banned_keywords=banned_keywords,
                      daily_carb_target=daily_carb_target,    # optional
                      daily_fat_target=daily_fat_target,      # optional
                      step=fitness_scenario.step,             # enables annealing
            )

            print(f"Nutrition reward: {reward}")
            print(f"Info: {nutri_info}")

            # Make provenance a bit softer early, stricter later
            if fitness_scenario.step < 100:
                tol_pct = 0.10        # allow 10% macro deviation early
                name_cutoff = 0.85    # easier fuzzy name match
            else:
                tol_pct = 0.05        # tighten to 5% later
                name_cutoff = 0.92    # stricter match

            prov_score, prov_info = provenance_reward_names_only_totals_only(
                payload,
                traj,
                tol_pct=tol_pct,
                name_match_cutoff=name_cutoff,
            )
            # prov_score, prov_info = provenance_reward_names_only_totals_only(
            #     payload,
            #     traj,
            #     tol_pct=tol_pct,
            #     name_match_cutoff=name_cutoff,
            # )

            # prov_score, prov_info = provenance_reward_names_only_totals_only(payload, traj, tol_pct=0.05)
            print(f"Provenance reward: {prov_score}")
            print(f"Info: {prov_info}")

            # # Combine (choose weights; example below)
            # W_MACROS_SCHEMA   = 0.7
            # W_PROVEN   = 0.3   # NEW: encourages using tool results faithfully

            # total_reward += (W_MACROS_SCHEMA * reward) + (W_PROVEN * prov_score)

            # Emphasize provenance slightly with an exponent beta > 1
            BETA = 1.5   # try 1.3–2.0
            GAMMA = 1.0  # macro emphasis; increase to >1 if you want tighter macro pressure

            _product_reward = (reward ** GAMMA) * (prov_score ** BETA)

            # This product is just a diagnostic metric now, not the RL reward.
            traj.metrics["macros_prov_product"] = _product_reward

            traj.metrics["macros_schema"] = reward
            traj.metrics["provenance"] = prov_score

            macros_schema_score = reward   # use the SAME metric you plotted for macros/schema
            provenance_score    = prov_score     # the one you plotted for provenance

            score, info = combine_reward_grpo(
                nutri=macros_schema_score,
                prov=provenance_score,
                step=getattr(fitness_scenario, "step", None),
                mix=0.25
                # gamma=1.0,      # usually fine
                # beta=None,      # let schedule drive it
                # eps=0.03, mix=0.15, jitter=1e-3  # default values
            )

            traj.reward = score
            print(f"Combined reward: {score}")
            print(f"Info: {info}")
            #traj.metrics["reward_info"] = info



            # traj.final_answer = final_answer
            # # Score the trajectory
            # correctness_judge_response = await nutrition_judge_score(
            #     scenario, traj.final_answer.answer
            # )

            # judge_score= correctness_judge_response.score
            # print("Judge score:", judge_score)
            # total_reward = total_reward + judge_score

            # judge_reasoning = correctness_judge_response.reasoning
            # print("Judge reasoning:", judge_reasoning)
            # traj.metrics["judge_reasoning"] = judge_reasoning



            random_noise = random.random()
            #traj.reward = total_reward + random_noise * 0.005
            print(f"Total reward: {traj.reward}")
            traj.metrics["correct"] = score
        else:
            # CRITICAL: Penalize trajectories that don't produce a final answer
            # Penalize trajectories that don't produce a final answer,
            # but keep the penalty mild so it doesn't dominate training.
            print("❌ No final answer produced!")
            traj.reward = -0.1
            traj.metadata["failure_reason"] = "no_final_answer"
            traj.metrics["failure_code"] = 0.5
            traj.metrics["correct"] = 0.0
            traj.metrics["macros_schema"] = 0.0
            traj.metrics["provenance"] = 0.0


    except Exception as e:
        print(f"❌ Error running LangGraph agent: {e}")
        
        # Different penalties for different error types (milder)
        error_str = str(e).lower()
        failure_reason = "unknown_error"
        failure_code = 2.0
        
        if "timeout" in error_str:
            traj.reward = -0.05      # small penalty
            failure_reason = "timeout"
            failure_code = 3.0
        elif "parse" in error_str or "json" in error_str:
            traj.reward = -0.08      # slightly stronger, but still mild
            failure_reason = "parse_error"
            failure_code = 4.0
        else:
            traj.reward = -0.1       # generic failure
        
        traj.metadata["failure_reason"] = failure_reason
        traj.metrics["failure_code"] = failure_code
        traj.metrics["correct"] = 0.0
        traj.metrics["macros_schema"] = 0.0
        traj.metrics["provenance"] = 0.0
        
        traj.messages_and_choices.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )

    return traj


print("LangGraph rollout function defined!")


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

async def run_validation(
    model: art.Model,
    val_scenarios: List[Scenario],
    step: int,
    num_samples: int = 10
) -> Dict[str, float]:
    """
    Run validation on a subset of validation scenarios.
    
    Args:
        model: The model to evaluate
        val_scenarios: List of validation scenarios
        step: Current training step
        num_samples: Number of validation samples to evaluate
    
    Returns:
        Dictionary with validation metrics
    """
    print(f"\n{'='*80}")
    print(f"🔍 Running Validation at Step {step}")
    print(f"{'='*80}")
    
    val_rewards = []
    val_nutrition_scores = []
    val_provenance_scores = []
    val_success_rate = []
    
    # Take a subset of validation scenarios
    val_sample = val_scenarios[:num_samples]
    
    for idx, val_scenario in enumerate(val_sample, 1):
        print(f"\nValidating {idx}/{len(val_sample)}: {val_scenario.id}")
        
        try:
            # Run rollout
            val_traj = await rollout(
                model, 
                FitnessScenario(step=step, scenario=val_scenario)
            )
            
            # Collect metrics
            val_rewards.append(val_traj.reward)
            
            # Extract individual component scores
            if hasattr(val_traj, 'metrics'):
                if 'macros_schema' in val_traj.metrics:
                    val_nutrition_scores.append(val_traj.metrics['macros_schema'])
                if 'provenance' in val_traj.metrics:
                    val_provenance_scores.append(val_traj.metrics['provenance'])
            
            # Track success (has final answer)
            success = 1.0 if val_traj.final_answer is not None else 0.0
            val_success_rate.append(success)
            
            print(f"  Reward: {val_traj.reward:.4f}, Success: {success}")
            
        except Exception as e:
            print(f"  ⚠️ Validation error: {e}")
            val_rewards.append(-0.5)
            val_success_rate.append(0.0)
    
    # Calculate aggregate metrics
    metrics = {
        "val/reward_mean": sum(val_rewards) / len(val_rewards) if val_rewards else 0.0,
        "val/reward_std": (sum((r - sum(val_rewards)/len(val_rewards))**2 for r in val_rewards) / len(val_rewards))**0.5 if len(val_rewards) > 1 else 0.0,
        "val/success_rate": sum(val_success_rate) / len(val_success_rate) if val_success_rate else 0.0,
    }
    
    if val_nutrition_scores:
        metrics["val/nutrition_mean"] = sum(val_nutrition_scores) / len(val_nutrition_scores)
    
    if val_provenance_scores:
        metrics["val/provenance_mean"] = sum(val_provenance_scores) / len(val_provenance_scores)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 Validation Results (Step {step}):")
    print(f"{'='*80}")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print(f"{'='*80}\n")
    
    return metrics


async def main():
    """Main training loop"""
    global model, backend
    
    print("=" * 80)
    print("Starting Fitness Agent RAG Training")
    print("=" * 80)
    
    # Initialize model and backend
    print("\n🔧 Setting up model and backend...")
    random.seed(42)
    
    model = art.TrainableModel(
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL_NAME,
    )
    
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=8192,
        ),
        engine_args=art.dev.EngineArgs(
            enforce_eager=True,
            gpu_memory_utilization=0.85,  # B200 has massive VRAM (192GB) - use more!
        ),
    )
    
    # Initialize the backend
    backend = LocalBackend(
        in_process=True,
        path="./.art",
    )
    
    # Register the model with the backend
    print("📝 Registering model with backend...")
    await model.register(backend)
    print("✅ Model registered successfully!")
    
    # Initialize Weave for tracking
    if os.getenv("WANDB_API_KEY", ""):
        print("📊 Initializing Weave for experiment tracking...")
        weave.init(model.project, settings={"print_call_link": False})
        print("✅ Weave initialized!")
    
    # Training configuration
    from art.utils import iterate_dataset
    from art.langgraph import wrap_rollout

    training_config = {
        "groups_per_step": 6,
        "num_epochs": 3,
        "rollouts_per_group": 12,
        "learning_rate": 1e-5,
        "max_steps": 150,
        "validation_every": 5,      # Run validation every N steps
        "validation_samples": 10,     # Number of validation samples
        "save_best": True,            # Save best model based on validation
    }

    print(f"\n📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Split scenarios into train and validation sets
    train_scenarios = [s for s in scenarios_list if s.split == "train"]
    val_scenarios = [s for s in scenarios_list if s.split == "test"]
    
    print(f"\n📊 Dataset Split:")
    print(f"  Training scenarios: {len(train_scenarios)}")
    print(f"  Validation scenarios: {len(val_scenarios)}")
    
    # If no explicit validation set, create one from training
    if not val_scenarios:
        print("  ⚠️ No validation scenarios found, creating 80/20 split from training data")
        random.shuffle(train_scenarios)
        split_idx = int(len(train_scenarios) * 0.8)
        val_scenarios = train_scenarios[split_idx:]
        train_scenarios = train_scenarios[:split_idx]
        print(f"  New split - Train: {len(train_scenarios)}, Val: {len(val_scenarios)}")
    
    print()

    # Use iterate_dataset with training scenarios only
    training_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
    )
    
    # Track best validation performance
    best_val_reward = float('-inf')
    best_val_step = 0

    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")

        # Create trajectory groups for this batch
        groups = []
        for scenario_data in batch.items:
            scenario = scenario_data
            print(scenario)
            groups.append(
                art.TrajectoryGroup(
                    (
                        wrap_rollout(model, rollout)(
                            model, FitnessScenario(step=batch.step, scenario=scenario.model_dump())
                        )
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )
        print("Group:", groups[0])
        
        # Gather all trajectory groups
        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        print("Finished groups:", finished_groups)

        # Train on the gathered trajectories
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            # Lowering the logprob_calculation_chunk_size is a memory saving measure
            # to allow longer sequences (up to 8192 tokens) to be processed on a T4.
            _config={"logprob_calculation_chunk_size": 8},
        )

        print(f"✅ Completed training step {batch.step}")
        
        # Run validation periodically
        should_validate = (
            batch.step % training_config["validation_every"] == 0 or
            batch.step == training_config["max_steps"] or
            batch.step == 1  # Always validate after first step
        )
        
        if should_validate and val_scenarios:
            val_metrics = await run_validation(
                model=model,
                val_scenarios=val_scenarios,
                step=batch.step,
                num_samples=training_config["validation_samples"]
            )
            
            # Log to Weave/W&B if available
            if os.getenv("WANDB_API_KEY", ""):
                try:
                    import wandb
                    # Log validation metrics
                    wandb.log({
                        "step": batch.step,
                        **val_metrics
                    })
                except Exception as e:
                    print(f"⚠️ Failed to log to W&B: {e}")
            
            # Track best model
            current_val_reward = val_metrics.get("val/reward_mean", float('-inf'))
            if training_config["save_best"] and current_val_reward > best_val_reward:
                best_val_reward = current_val_reward
                best_val_step = batch.step
                print(f"\n🌟 New best validation reward: {best_val_reward:.4f} at step {batch.step}")
                
                # Save checkpoint (optional - ART handles this automatically)
                # You can add custom checkpoint saving here if needed
        
        # Stop after max_steps
        if batch.step >= training_config["max_steps"]:
            print(f"\n🎉 Training complete! Reached max_steps: {training_config['max_steps']}")
            if best_val_reward > float('-inf'):
                print(f"🏆 Best validation reward: {best_val_reward:.4f} at step {best_val_step}")
            break

    print("\n" + "=" * 80)
    print("Training finished successfully!")
    print("=" * 80)
    print(f"\n📈 Final Statistics:")
    print(f"  Total steps: {batch.step}")
    if best_val_reward > float('-inf'):
        print(f"  Best validation reward: {best_val_reward:.4f} (step {best_val_step})")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

