#!/usr/bin/env python3
"""
RAG-FitnessRL Benchmark Script

Benchmarks different LLMs (GPT-4o-mini, GPT-4, Claude, etc.) on the fitness agent task
using the same LangGraph ReAct agent setup and evaluation metrics.

PREREQUISITES:
--------------
1. Install dependencies: `uv sync`
2. Set up environment variables in .env file:
   - OPENAI_API_KEY (required for OpenAI models)
   - ANTHROPIC_API_KEY (required for Claude models)
   - PINECONE_API_KEY (required)
   - WANDB_API_KEY (optional, for logging)

USAGE:
------
    python scripts/becnhmark_rag_fitness_art.py

The script will:
1. Load test scenarios from data/fitness_scenarios.jsonl
2. Test each configured LLM with the same agent setup
3. Evaluate using the same reward functions (nutrition, provenance)
4. Generate comparison reports

REQUIREMENTS:
-------------
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
from collections import defaultdict

# Third-party imports
from dotenv import load_dotenv
from datasets import load_dataset
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

# LangChain and LangGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# LiteLLM
from litellm import acompletion

# Pinecone
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Ensure we're in the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Validate required environment variables
if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is required. Please set it in your .env file")

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
class BenchmarkTrajectory:
    """Trajectory for benchmarking (no ART dependency)"""
    model_name: str
    scenario_id: str
    final_answer: FinalAnswer | None = None
    messages_and_choices: List[Dict[str, Any]] = None
    metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.messages_and_choices is None:
            self.messages_and_choices = []
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# DATA LOADING
# ============================================================================

# Load the dataset from the JSONL file
dataset_path = project_root / "data" / "fitness_scenarios.jsonl"
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

dataset = load_dataset("json", data_files=str(dataset_path))
training_scenarios = dataset["train"]

print("Dataset loaded successfully!")
print(training_scenarios)

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

training_scenarios = training_scenarios.map(one_day_meal_question)
training_scenarios = training_scenarios.map(combine_question_and_context)
training_scenarios = training_scenarios.map(convert_val_to_test)
training_scenarios = training_scenarios.map(get_target_nutrition_data)
training_scenarios = training_scenarios.map(extract_context_columns)

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

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Models to benchmark
MODELS_TO_TEST = [
    {"name": "gpt-4o-mini", "provider": "openai", "temperature": 0.2},
    {"name": "gpt-4o", "provider": "openai", "temperature": 0.2},
    {"name": "gpt-4-turbo", "provider": "openai", "temperature": 0.2},
    {"name": "claude-3-5-sonnet-20241022", "provider": "anthropic", "temperature": 0.2},
    {"name": "claude-3-opus-20240229", "provider": "anthropic", "temperature": 0.2},
]

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
    return [{"id" : hit['_id'] ,"name": hit["fields"]["name"], "calories":  hit["fields"]["calories"],"carbs":  hit["fields"]["carbs"], "protein": hit["fields"]["proteins"], "fat":  hit["fields"]["fats"]}  for hit in data["hits"] if "fields" in hit and "name" in hit["fields"]]

# ============================================================================
# PAYLOAD PARSING UTILITIES (same as training script)
# ============================================================================

def _extract_first_json_segment(s: str) -> str | None:
    """Extract the first complete top-level JSON object or array from a noisy string."""
    start_candidates = [s.find('{'), s.find('[')]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)

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

            if want_obj and depth_obj == 0 and depth_arr == 0 and i >= start:
                return s[start:i+1]
            if want_arr and depth_arr == 0 and depth_obj == 0 and i >= start:
                return s[start:i+1]
    return None

def _loads_loose(s: str):
    """Try multiple strategies to convert a string into JSON."""
    try:
        v = json.loads(s)
    except json.JSONDecodeError:
        v = None

    tries = 0
    while isinstance(v, str) and tries < 3:
        tries += 1
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            break

    if v is not None and not isinstance(v, str):
        return v

    seg = _extract_first_json_segment(s)
    if seg is not None:
        try:
            return json.loads(seg)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not parse JSON from string", s, 0)

def get_payload(obj):
    """Return a dict payload from FinalAnswer/str/dict."""
    if hasattr(obj, "answer"):
        obj = obj.answer

    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8", errors="replace")

    if isinstance(obj, str):
        try:
            obj = _loads_loose(obj)
        except json.JSONDecodeError:
            return {"_error": "invalid_json_string", "_raw": obj}

    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "dict"):
        try:
            obj = obj.dict()
        except TypeError:
            pass

    if is_dataclass(obj):
        obj = asdict(obj)

    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {"_root": obj}

    return {"_error": f"unexpected_type:{type(obj).__name__}", "_raw": str(obj)}

# ============================================================================
# PROVENANCE REWARD FUNCTION (same as training script)
# ============================================================================

def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _rel_err(true, pred):
    if true == 0:
        return 0.0 if abs(pred) < 1e-6 else 1.0
    return abs(pred - true) / max(1.0, abs(true))

def _infer_multiplier(unit: dict, totals: dict, min_dims=2):
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
    meals = []
    if isinstance(payload, dict) and isinstance(payload.get("meals"), list):
        for m in payload["meals"]:
            if isinstance(m, dict):
                meals.append(m)
    return meals

def _extract_totals_from_meal(meal: dict):
    totals = {}
    quantity = None
    
    if not isinstance(meal, dict):
        return totals, quantity
    
    if "quantity" in meal:
        try:
            quantity = float(meal["quantity"])
        except (ValueError, TypeError):
            quantity = None
    
    if "calories" in meal: totals["calories"] = meal["calories"]
    if "carbs"    in meal: totals["carbs"]    = meal["carbs"]
    if "proteins" in meal: totals["protein"]  = meal["proteins"]
    if "fats"     in meal: totals["fat"]      = meal["fats"]
    
    return totals, quantity

def _collect_catalog_from_logs_by_name(traj, tool_name="recipe_semantic_search"):
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
                continue
    return catalog

def provenance_reward_names_only_totals_only(
    payload: dict,
    traj,
    tol_pct: float = 0.05,
    name_match_cutoff: float = 0.92,
    tool_name: str = "recipe_semantic_search",
):
    catalog = _collect_catalog_from_logs_by_name(traj, tool_name=tool_name)
    if not catalog:
        return 0.0, {"reason": "no_recipe_tool_usage_detected"}

    meals = _flatten_plan_meals(payload)
    if not meals:
        return 0.0, {"reason": "empty_plan"}

    checked = passed = 0
    details = []

    cat_names = list(catalog.keys())

    for meal in meals:
        checked += 1
        plan_name_raw = meal.get("name", "")
        plan_key = _norm_name(plan_name_raw)

        match_key = plan_key if plan_key in catalog else None

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

        if explicit_quantity is not None:
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

from src.env.verifiable_rewards.nutrition_rewards import nutrition_reward

# ============================================================================
# BENCHMARK ROLLOUT FUNCTION
# ============================================================================

async def benchmark_rollout(
    chat_model: BaseChatModel,
    model_name: str,
    scenario: Scenario,
) -> BenchmarkTrajectory:
    """Run a single benchmark rollout with a given LLM model."""
    
    traj = BenchmarkTrajectory(
        model_name=model_name,
        scenario_id=str(scenario.id),
        metadata={
            "scenario_id": str(scenario.id),
        },
    )

    system_prompt = dedent(PLANNER_PROMPT)
    final_answer = None

    daily_cal_target = scenario.daily_cal_target
    daily_prot_target = scenario.daily_prot_target
    daily_carb_target = scenario.daily_carb_target
    daily_fat_target = scenario.daily_fat_target

    def log_tool(tool_name):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                call = {
                    "tool": tool_name,
                    "ts": datetime.utcnow().isoformat(),
                    "args": args,
                    "kwargs": kwargs
                }

                traj.messages_and_choices.append(
                    {"role": "tool_log", "content": json.dumps({"start": call})}
                )

                try:
                    out = fn(*args, **kwargs)
                    traj.messages_and_choices.append(
                        {"role": "tool_log", "content": json.dumps({"end": {**call, 'result': out}})}
                    )
                    return out
                except Exception as e:
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
      return results

    @tool
    @log_tool("return_final_answer_tool")
    def return_final_answer_tool(answer: str) -> dict:
        """Return the final answer (daily meal plan) in the correct format """
        nonlocal final_answer
        payload = get_payload(answer)
        final_answer = FinalAnswer(answer=payload)
        return final_answer.model_dump()

    tools = [recipe_semantic_search, return_final_answer_tool]

    # Create the LangGraph ReAct agent
    react_agent = create_react_agent(chat_model, tools)

    try:
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": MAX_TURNS,
        }

        res = await react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=scenario.question),
                ]
            },
            config=config,
        )

        if final_answer:
            payload = get_payload(final_answer)
            traj.final_answer = final_answer

            banned_keywords = scenario.banned_keywords

            reward, nutri_info = nutrition_reward(
                payload,
                daily_cal_target=daily_cal_target,
                daily_prot_target=daily_prot_target,
                banned_keywords=banned_keywords,
                daily_carb_target=daily_carb_target,
                daily_fat_target=daily_fat_target,
                step=None,  # No step for benchmarking
            )

            prov_score, prov_info = provenance_reward_names_only_totals_only(
                payload,
                traj,
                tol_pct=0.05,
                name_match_cutoff=0.92,
            )

            traj.metrics["macros_schema"] = reward
            traj.metrics["provenance"] = prov_score
            traj.metrics["schema_reward"] = nutri_info.get("R_schema", 0.0)
            traj.metrics["macro_reward"] = nutri_info.get("R_macro", 0.0)
            traj.metrics["banned_reward"] = nutri_info.get("R_banned", 0.0)
            
            is_strict_success = (
                nutri_info.get("R_schema", 0.0) == 1.0 and
                nutri_info.get("R_macro", 0.0) > 0.99 and
                nutri_info.get("R_banned", 1.0) == 1.0 and
                prov_score == 1.0
            )
            traj.metrics["strict_success"] = 1.0 if is_strict_success else 0.0

            # Combined score (same as training)
            score, info = combine_reward_grpo(
                nutri=reward,
                prov=prov_score,
                step=None,
                mix=0.25
            )
            traj.metrics["combined_score"] = score
        else:
            traj.metrics["combined_score"] = 0.0
            traj.metrics["macros_schema"] = 0.0
            traj.metrics["provenance"] = 0.0
            traj.metrics["strict_success"] = 0.0
            traj.metadata["failure_reason"] = "no_final_answer"

    except Exception as e:
        traj.metrics["combined_score"] = 0.0
        traj.metrics["macros_schema"] = 0.0
        traj.metrics["provenance"] = 0.0
        traj.metrics["strict_success"] = 0.0
        traj.metadata["failure_reason"] = str(e)
        traj.messages_and_choices.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )

    return traj

def combine_reward_grpo(
    nutri: float,
    prov: float,
    step: int | None = None,
    gamma: float = 1.0,
    beta: float | None = None,
    eps: float = 0.03,
    mix: float = 0.25,
    jitter: float = 1e-3,
) -> Tuple[float, Dict[str, Any]]:
    """Combined reward function (same as training script)."""
    def _beta_schedule(s: int | None) -> float:
        if s is None:
            return 1.0
        if s < 50:
            return 0.5
        if s < 150:
            return 1.0
        if s < 300:
            return 1.5
        return 1.8

    B = beta if beta is not None else _beta_schedule(step)

    n = max(0.0, min(1.0, float(nutri)))
    p = max(0.0, min(1.0, float(prov)))
    n_gate = max(eps, n)
    p_gate = max(eps, p)

    gate = (n_gate ** gamma) * (p_gate ** B)
    gate_min = (eps ** (gamma + B))
    if gate_min < 1.0:
        gate = (gate - gate_min) / (1.0 - gate_min)
    gate = max(0.0, min(1.0, gate))

    dense = 0.6 * p + 0.4 * n
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
# BENCHMARK MAIN FUNCTION
# ============================================================================

def create_chat_model(model_config: Dict[str, Any]) -> BaseChatModel:
    """Create a LangChain chat model from config."""
    provider = model_config["provider"]
    name = model_config["name"]
    temperature = model_config.get("temperature", 0.2)
    
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(f"OPENAI_API_KEY required for {name}")
        return ChatOpenAI(model=name, temperature=temperature)
    elif provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError(f"ANTHROPIC_API_KEY required for {name}")
        return ChatAnthropic(model=name, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")

async def run_benchmark():
    """Run benchmarks across all configured models."""
    
    print("=" * 80)
    print("Starting LLM Benchmark")
    print("=" * 80)
    
    # Get test scenarios
    test_scenarios = [s for s in scenarios_list if s.split == "test"]
    if not test_scenarios:
        print("⚠️ No test scenarios found, using first 20 training scenarios")
        test_scenarios = scenarios_list[:20]

    test_scenarios = test_scenarios[:20]

    # Set seed for reproducibility
    set_all_seeds(SEED)
    
    print(f"\n📊 Benchmarking on {len(test_scenarios)} scenarios")
    print(f"📋 Models to test: {[m['name'] for m in MODELS_TO_TEST]}\n")
    
    # Results storage
    all_results = defaultdict(list)
    
    # Test each model
    for model_config in MODELS_TO_TEST:
        model_name = model_config["name"]
        print(f"\n{'='*80}")
        print(f"🧪 Testing Model: {model_name}")
        print(f"{'='*80}")
        
        try:
            # Create chat model
            chat_model = create_chat_model(model_config)
            
            # Run benchmarks
            for idx, scenario in enumerate(tqdm(test_scenarios, desc=f"Benchmarking {model_name}")):
                try:
                    traj = await benchmark_rollout(
                        chat_model=chat_model,
                        model_name=model_name,
                        scenario=scenario,
                    )
                    all_results[model_name].append(traj)
                except Exception as e:
                    print(f"\n⚠️ Error on scenario {scenario.id}: {e}")
                    # Create a failed trajectory
                    failed_traj = BenchmarkTrajectory(
                        model_name=model_name,
                        scenario_id=str(scenario.id),
                        metrics={"combined_score": 0.0, "macros_schema": 0.0, "provenance": 0.0, "strict_success": 0.0},
                        metadata={"failure_reason": str(e)}
                    )
                    all_results[model_name].append(failed_traj)
        
        except Exception as e:
            print(f"\n❌ Failed to initialize model {model_name}: {e}")
            continue
    
    # Aggregate and report results
    print("\n" + "=" * 80)
    print("📊 BENCHMARK RESULTS")
    print("=" * 80)
    
    summary = {}
    for model_name, trajectories in all_results.items():
        if not trajectories:
            continue
            
        metrics_agg = defaultdict(list)
        for traj in trajectories:
            for metric_name, metric_value in traj.metrics.items():
                metrics_agg[metric_name].append(metric_value)
        
        summary[model_name] = {
            metric: {
                "mean": sum(values) / len(values),
                "std": (sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5 if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
            for metric, values in metrics_agg.items()
        }
        
        # Print per-model summary
        print(f"\n{model_name}:")
        print(f"  Scenarios tested: {len(trajectories)}")
        for metric_name, stats in summary[model_name].items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f})")
    
    # Comparison table
    print("\n" + "=" * 80)
    print("📈 COMPARISON TABLE")
    print("=" * 80)
    
    if summary:
        # Get all metrics
        all_metrics = set()
        for model_stats in summary.values():
            all_metrics.update(model_stats.keys())
        
        # Print header
        print(f"\n{'Model':<30}", end="")
        for metric in sorted(all_metrics):
            print(f"{metric:<20}", end="")
        print()
        print("-" * (30 + 20 * len(all_metrics)))
        
        # Print rows
        for model_name, model_stats in summary.items():
            print(f"{model_name:<30}", end="")
            for metric in sorted(all_metrics):
                if metric in model_stats:
                    print(f"{model_stats[metric]['mean']:.4f} ± {model_stats[metric]['std']:.4f}  ", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()
    
    # Save detailed results
    results_file = project_root / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": {k: v for k, v in summary.items()},
            "detailed": {
                model_name: [
                    {
                        "scenario_id": traj.scenario_id,
                        "metrics": traj.metrics,
                        "metadata": traj.metadata,
                    }
                    for traj in trajectories
                ]
                for model_name, trajectories in all_results.items()
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_benchmark())
